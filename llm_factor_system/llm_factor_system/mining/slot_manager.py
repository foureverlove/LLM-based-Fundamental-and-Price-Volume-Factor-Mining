# ==================== mining/slot_manager.py ====================
"""
多槽位并行管理器
"""
import threading
import queue
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import sys

from factor.factor import Factor
from factor.selector import ImprovedMMRSelector
from llm.generator import LLMFactorGenerator
from dsl.executor import DSLExecutor
from dsl.expression_corrector import ExpressionCorrector
from factor.evaluator import FactorEvaluator

class MiningSlot:
    """单个挖掘槽位"""
    
    def __init__(self, slot_id: int, config: Dict[str, Any], 
                 shared_resources: Dict[str, Any]):
        self.slot_id = slot_id
        self.config = config
        self.shared = shared_resources
        
        # 私有资源
        # 过滤掉LLMFactorGenerator不接受的参数（如timeout）
        llm_config = config.get('llm', {}).copy()
        llm_config.pop('timeout', None)  # 移除timeout参数
        self.generator = LLMFactorGenerator(**llm_config)
        self.executor = DSLExecutor()
        self.evaluator = FactorEvaluator()
        self.selector = ImprovedMMRSelector(
            lambda_param=config['mining']['mmr_lambda'],
            barra_factors=shared_resources.get('barra_factors', {})
        )
        
        # 表达式修正器
        operator_lib = self.executor.operators
        field_info = {
            field: {'type': 'price', 'frequency': 'daily'} 
            for field in shared_resources['train_data'].keys()
        }
        self.corrector = ExpressionCorrector(operator_lib, field_info)
        # logger
        self.logger = logging.getLogger(f"mining.slot.{self.slot_id}")
        
        # 状态
        self.is_running = False
        self.current_factor = None
        self.generated_count = 0
        self.accepted_count = 0
        
        # 数据副本（避免并发冲突）
        self.train_data = shared_resources['train_data'].copy()
        self.train_returns = shared_resources['train_returns'].copy()
    
    def run_one_iteration(self) -> Optional[Factor]:
        """运行一次挖掘迭代"""
        retry_times = self.config['mining'].get('retry_times', 3)
        
        for attempt in range(retry_times):
            try:
                # 1. 生成因子表达式
                expr, explanation = self._generate_expression()
                if not expr:
                    self.logger.debug(f"Slot {self.slot_id} attempt {attempt}: empty expression, skipping")
                    continue
                
                # 1.5. 修正表达式
                try:
                    expr = self.corrector.correct(expr)
                except Exception as e:
                    # 修正失败，使用原始表达式
                    self.logger.debug(f"Slot {self.slot_id} attempt {attempt}: expression correction failed: {e}")
                    pass
                
                # 2. 执行表达式计算因子值
                factor_values = self.executor.run(expr, self.train_data)
                if factor_values is None or factor_values.empty:
                    self.logger.debug(f"Slot {self.slot_id} attempt {attempt}: factor_values empty or None for expr={expr}")
                    continue
                
                # 检查数据有效性
                if factor_values.isna().all().all():
                    self.logger.debug(f"Slot {self.slot_id} attempt {attempt}: factor_values all NaN for expr={expr}")
                    continue
                
                # 3. 评估因子表现
                ic_mean, ic_ts = self.evaluator.calc_ic(factor_values, self.train_returns)
                pnl = self.evaluator.simple_long_short(factor_values, self.train_returns)
                
                # 检查IC有效性
                if abs(ic_mean) < 1e-6:
                    self.logger.debug(f"Slot {self.slot_id} attempt {attempt}: ic_mean too small ({ic_mean}) for expr={expr}")
                    continue
                
                # 4. 创建因子对象
                factor = Factor(
                    expr=expr,
                    explanation=explanation,
                    values=factor_values,
                    ic_mean=ic_mean,
                    ic_ts=ic_ts,
                    pnl=pnl
                )
                
                # 5. 相关性筛选
                selected_factors = self.shared['factor_pool'].get_all_factors()
                score = self.selector.score(factor, selected_factors)
                
                # 6. 检查是否达到入库标准
                if (abs(ic_mean) >= self.config['mining']['ic_threshold'] and 
                    score > 0):
                    
                    # 样本外验证
                    test_result = self._validate_out_of_sample(factor)
                    if test_result['passed']:
                        factor.test_metrics = test_result

                        # 线程安全地添加到共享因子库
                        with self.shared['factor_pool'].lock:
                            added = self.shared['factor_pool'].add_factor(factor)

                        if added:
                            self.accepted_count += 1
                            self.logger.info(f"Slot {self.slot_id} accepted factor expr={expr} ic={ic_mean:.4f} score={score:.4f} test_return={test_result.get('test_return')}")
                        else:
                            self.logger.info(f"Slot {self.slot_id} factor rejected as duplicate when adding expr={expr}")
                        
                        # 提取改进想法
                        try:
                            idea = self.generator.extract_idea(expr, {
                                'ic_mean': ic_mean,
                                'score': score
                            })
                            # 添加到共享想法池
                            with self.shared['idea_pool'].lock:
                                self.shared['idea_pool'].add_idea(idea)
                        except Exception as e:
                            # Idea提取失败不影响因子入库
                            self.logger.debug(f"Slot {self.slot_id}: idea extraction failed: {e}")
                
                self.generated_count += 1
                return factor
                
            except Exception as e:
                self.logger.exception(f"Slot {self.slot_id} attempt {attempt} exception: {e}")
                if attempt == retry_times - 1:
                    self.logger.error(f"Slot {self.slot_id} error after {retry_times} attempts: {e}")
                continue
        
        return None
    
    def _generate_expression(self) -> Tuple[str, str]:
        """生成因子表达式（带有RAG启发）"""
        # 从想法池获取启发
        ideas = self.shared['idea_pool'].get_random_ideas(2)
        
        # 从RAG获取相似因子
        rag_context = []
        if self.shared.get('rag_handler'):
            similar_factors = self.shared['rag_handler'].search(
                query="量价因子", top_k=3
            )
            rag_context = [f.expr for f in similar_factors]
        
        # 根据概率选择生成策略
        import random
        # allow configurable improvement probability
        improvement_prob = self.config['mining'].get('improvement_prob', 0.2)
        rag_prob = self.config['mining'].get('rag_prob', 0.4)
        random_prob = max(0.0, 1.0 - improvement_prob - rag_prob)
        strategy = random.choices(
            ['random', 'rag', 'improvement'],
            weights=[random_prob, rag_prob, improvement_prob]
        )[0]
        
        if strategy == 'rag' and rag_context:
            return self.generator.generate_price_factor(
                available_fields=list(self.train_data.keys()),
                rag_context=rag_context
            )
        elif strategy == 'improvement' and ideas:
            # 随机选择一个现有因子进行改进
            existing_factors = self.shared['factor_pool'].get_all_factors()
            if existing_factors:
                target_factor = random.choice(existing_factors)
                return self.generator.improve_factor(
                    original_expr=target_factor.expr,
                    original_metrics={'ic_mean': target_factor.ic_mean},
                    available_fields=list(self.train_data.keys()),
                    improvement_idea=random.choice(ideas)
                )
        
        # 随机生成
        return self.generator.generate_price_factor(
            available_fields=list(self.train_data.keys())
        )
    
    def _validate_out_of_sample(self, factor: Factor) -> Dict[str, Any]:
        """样本外验证"""
        test_data = self.shared['test_data']
        test_returns = self.shared['test_returns']
        
        try:
            # 计算样本外因子值
            test_values = self.executor.run(factor.expr, test_data)
            if test_values is None or test_values.empty:
                return {'passed': False, 'reason': '计算失败'}
            
            # 计算样本外IC
            test_ic_mean, _ = self.evaluator.calc_ic(test_values, test_returns)
            
            # 计算样本外收益
            test_pnl = self.evaluator.simple_long_short(test_values, test_returns)
            test_return = test_pnl.iloc[-1] / len(test_pnl) * 252 if len(test_pnl) > 0 else 0
            
            # 验证标准
            passed = (
                abs(test_ic_mean) >= self.config['mining']['ic_threshold'] * 0.8 and
                test_return >= self.config['mining']['excess_return_threshold'] * 0.8
            )
            
            return {
                'passed': passed,
                'test_ic_mean': test_ic_mean,
                'test_return': test_return,
                'test_pnl': test_pnl
            }
            
        except Exception as e:
            return {'passed': False, 'reason': str(e)}

class SlotManager:
    """多槽位管理器"""
    
    def __init__(self, config: Dict[str, Any], shared_resources: Dict[str, Any]):
        self.config = config
        self.shared = shared_resources
        
        # 创建槽位
        self.slots = [
            MiningSlot(i, config, shared_resources) 
            for i in range(config['mining']['max_parallel_slots'])
        ]
        
        # 控制变量
        self.is_running = False
        self.total_generated = 0
        self.total_accepted = 0
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=config['mining']['max_parallel_slots'])
    
    def start(self):
        """启动所有槽位"""
        self.is_running = True
        
        # 读取终止目标
        target_count = self.config['mining'].get('target_count', 100)
        max_generations = self.config['mining'].get('max_generations', 1000)

        def _render_progress(accepted, generated):
            # 显示 Accepted / target_count 和 Generated / max_generations
            try:
                term_width = 60
                bar_width = 30
                pct = min(1.0, accepted / max(1, target_count))
                filled = int(bar_width * pct)
                bar = '#' * filled + '-' * (bar_width - filled)
                s = f"Accepted: {accepted}/{target_count} [{bar}] {pct*100:5.1f}%  | Generated: {generated}/{max_generations}"
                sys.stdout.write('\r' + s)
                sys.stdout.flush()
            except Exception:
                pass

        while self.is_running:
            futures = []
            for slot in self.slots:
                future = self.executor.submit(slot.run_one_iteration)
                futures.append(future)
            
            # 等待所有槽位完成
            for future in as_completed(futures):
                result = future.result()
                if result:
                    self.total_generated += 1
                    tm = getattr(result, 'test_metrics', None)
                    if isinstance(tm, dict) and tm.get('passed'):
                        self.total_accepted += 1
            
            # 输出进度（含控制台进度条）
            _render_progress(self.total_accepted, self.total_generated)
            
            # 检查停止条件
            if (self.total_accepted >= target_count or
                self.total_generated >= max_generations):
                # 完成，换行并停止
                try:
                    sys.stdout.write('\n')
                    sys.stdout.flush()
                except: pass
                self.stop()
                break
            
            # 短暂休眠
            time.sleep(1)
    
    def stop(self):
        """停止所有槽位"""
        self.is_running = False
        self.executor.shutdown(wait=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        slot_stats = []
        for slot in self.slots:
            slot_stats.append({
                'slot_id': slot.slot_id,
                'generated': slot.generated_count,
                'accepted': slot.accepted_count,
                'accept_rate': slot.accepted_count / max(slot.generated_count, 1)
            })
        
        return {
            'total_generated': self.total_generated,
            'total_accepted': self.total_accepted,
            'accept_rate': self.total_accepted / max(self.total_generated, 1),
            'slot_stats': slot_stats
        }