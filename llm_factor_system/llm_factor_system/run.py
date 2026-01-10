# ==================== run_7x24.py ====================

import time
import schedule
from datetime import datetime, timedelta
import logging
# Reduce noisy third-party logs early so progress bar isn't overwritten
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json

from config import Config
from data.minute_loader import MinuteDataLoader
from data.daily_loader import DailyDataLoader
from data.barra_loader import BarraLoader
from factor.factor_pool import FactorPool
from llm.idea_pool import IdeaPool
from llm.rag_handler import RAGHandler
from mining.slot_manager import SlotManager
from factor.synthesizer import FactorSynthesizer

class AlphaMiningSystem:
    """Alpha因子挖掘系统"""
    
    def __init__(self, config_path: str = None):
        # 加载配置
        self.config = Config(config_path)
        
        # 设置日志
        self._setup_logging()
        
        # 初始化组件
        self.logger.info("初始化Alpha因子挖掘系统...")
        
        # 数据加载器
        self.minute_loader = MinuteDataLoader(self.config.get("data_paths.minute"))
        self.daily_loader = DailyDataLoader(self.config.get("data_paths.daily"))
        self.barra_loader = BarraLoader(self.config.get("data_paths.barra"))
        
        # 加载数据
        self._load_data()
        
        # 共享资源
        self.shared_resources = self._init_shared_resources()
        
        # 槽位管理器
        self.slot_manager = SlotManager(
            config=self.config.config,
            shared_resources=self.shared_resources
        )
        
        # 因子合成器
        self.synthesizer = FactorSynthesizer()
        
        self.logger.info("系统初始化完成")
    
    def _setup_logging(self):
        """设置日志"""
        import os
        # 创建必要的目录
        os.makedirs("logs", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        os.makedirs("progress", exist_ok=True)
        
        log_config = self.config.get("logging")
        log_file = log_config.get("file", "logs/alpha_mining.log")
        
        logging.basicConfig(
            level=getattr(logging, log_config.get("level", "INFO")),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_data(self):
        """加载所有数据"""
        self.logger.info("加载数据...")
        
        # 加载日频数据
        self.logger.info("加载日频数据...")
        self.close = self.daily_loader.load_field("close")
        self.vwap = self.daily_loader.load_field("vwap")
        self.volume = self.daily_loader.load_field("amount")  # 成交额
        self.high = self.daily_loader.load_field("high")
        self.low = self.daily_loader.load_field("low")
        self.open = self.daily_loader.load_field("open")
        
        # #region agent log
        import json
        log_path = r"c:\Users\qtx27\Desktop\llm_factor_system\.cursor\debug.log"
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"data-load","hypothesisId":"H1","location":"run.py:90","message":"数据加载完成","data":{"close_shape":list(self.close.shape) if hasattr(self.close,'shape') else None,"close_cols_type":str(type(self.close.columns)) if hasattr(self.close,'columns') else None,"close_index_type":str(type(self.close.index)) if hasattr(self.close,'index') else None},"timestamp":int(__import__('time').time()*1000)}) + "\n")
        except Exception as e:
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"data-load","hypothesisId":"H1","location":"run.py:90","message":"日志写入失败","data":{"error":str(e)},"timestamp":int(__import__('time').time()*1000)}) + "\n")
            except: pass
        # #endregion
        
        # 计算收益率（修复 FutureWarning）
        # 收盘价：使用 shift(-1) 表示使用当日信号在次日收盘后观察收益
        self.close_returns = self.close.pct_change().shift(-1)
        # 开盘价：通常当信号基于开盘价时，需要跳过到下一个可交易日开盘，使用 shift(-2)
        self.open_returns = self.open.pct_change().shift(-2)
        # 默认的收益率仍然指向收盘价收益（保持兼容性）
        self.returns = self.close_returns
        
        # 分割训练集和测试集
        train_start = self.config.get("time_ranges.train_start")
        train_end = self.config.get("time_ranges.train_end")
        test_start = self.config.get("time_ranges.test_start")
        test_end = self.config.get("time_ranges.test_end")
        
        # 训练数据（确保数据对齐）
        train_close = self.close.loc[train_start:train_end]
        train_vwap = self.vwap.loc[train_start:train_end]
        train_volume = self.volume.loc[train_start:train_end]
        train_high = self.high.loc[train_start:train_end]
        train_low = self.low.loc[train_start:train_end]
        train_open = self.open.loc[train_start:train_end]
        
        # #region agent log
        import json
        log_path = r"c:\Users\qtx27\Desktop\llm_factor_system\.cursor\debug.log"
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"column-intersection","hypothesisId":"H1","location":"run.py:109","message":"检查columns类型","data":{"close_cols_type":str(type(train_close.columns)),"close_cols_is_index":isinstance(train_close.columns, pd.Index),"close_cols_sample":list(train_close.columns[:5]) if len(train_close.columns) > 0 else [],"vwap_cols_type":str(type(train_vwap.columns)),"volume_cols_type":str(type(train_volume.columns))},"timestamp":int(__import__('time').time()*1000)}) + "\n")
        except Exception as e:
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"column-intersection","hypothesisId":"H1","location":"run.py:109","message":"日志写入异常","data":{"error":str(e)},"timestamp":int(__import__('time').time()*1000)}) + "\n")
            except: pass
        # #endregion
        
        # 对齐列（取共同股票）- 使用 intersection 方法而不是 & 操作符
        # 确保所有columns都是Index类型
        try:
            common_cols = train_close.columns.intersection(train_vwap.columns)
            common_cols = common_cols.intersection(train_volume.columns)
            common_cols = common_cols.intersection(train_high.columns)
            common_cols = common_cols.intersection(train_low.columns)
            common_cols = common_cols.intersection(train_open.columns)
        except Exception as e:
            # #region agent log
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({"sessionId":"debug-session","runId":"column-intersection","hypothesisId":"H1","location":"run.py:125","message":"intersection操作失败","data":{"error":str(e),"error_type":type(e).__name__},"timestamp":int(__import__('time').time()*1000)}) + "\n")
            except: pass
            # #endregion
            # 降级方案：使用set操作
            common_cols = set(train_close.columns) & set(train_vwap.columns) & set(train_volume.columns) & \
                         set(train_high.columns) & set(train_low.columns) & set(train_open.columns)
            common_cols = pd.Index(list(common_cols))
        
        # #region agent log
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({"sessionId":"debug-session","runId":"column-intersection","hypothesisId":"H1","location":"run.py:135","message":"列对齐完成","data":{"common_cols_count":len(common_cols),"common_cols_type":str(type(common_cols)),"common_cols_sample":list(common_cols[:5]) if len(common_cols) > 0 else []},"timestamp":int(__import__('time').time()*1000)}) + "\n")
        except: pass
        # #endregion
        
        self.train_data = {
            "Close": train_close[common_cols],
            "Vwap": train_vwap[common_cols],
            "Volume": train_volume[common_cols],
            "High": train_high[common_cols],
            "Low": train_low[common_cols],
            "Open": train_open[common_cols]
        }
        self.train_returns = self.returns.loc[train_start:train_end, common_cols]
        
        # 测试数据
        test_close = self.close.loc[test_start:test_end]
        test_vwap = self.vwap.loc[test_start:test_end]
        test_volume = self.volume.loc[test_start:test_end]
        test_high = self.high.loc[test_start:test_end]
        test_low = self.low.loc[test_start:test_end]
        test_open = self.open.loc[test_start:test_end]
        
        # 对齐列（使用 intersection 方法，带错误处理）
        try:
            test_common_cols = test_close.columns.intersection(test_vwap.columns)
            test_common_cols = test_common_cols.intersection(test_volume.columns)
            test_common_cols = test_common_cols.intersection(test_high.columns)
            test_common_cols = test_common_cols.intersection(test_low.columns)
            test_common_cols = test_common_cols.intersection(test_open.columns)
        except Exception:
            # 降级方案：使用set操作
            test_common_cols = set(test_close.columns) & set(test_vwap.columns) & set(test_volume.columns) & \
                              set(test_high.columns) & set(test_low.columns) & set(test_open.columns)
            test_common_cols = pd.Index(list(test_common_cols))
        
        self.test_data = {
            "Close": test_close[test_common_cols],
            "Vwap": test_vwap[test_common_cols],
            "Volume": test_volume[test_common_cols],
            "High": test_high[test_common_cols],
            "Low": test_low[test_common_cols],
            "Open": test_open[test_common_cols]
        }
        self.test_returns = self.returns.loc[test_start:test_end, test_common_cols]
        
        # 加载Barra因子
        self.logger.info("加载Barra风险因子...")
        self.barra_factors = self.barra_loader.load_all_factors()
        
        self.logger.info(f"数据加载完成: "
                        f"训练集{len(self.train_returns)}天, "
                        f"测试集{len(self.test_returns)}天")
    
    def _init_shared_resources(self) -> Dict[str, Any]:
        """初始化共享资源"""
        return {
            'train_data': self.train_data,
            'train_returns': self.train_returns,
            'test_data': self.test_data,
            'test_returns': self.test_returns,
            'barra_factors': self.barra_factors,
            'factor_pool': FactorPool(),
            'idea_pool': IdeaPool(),
            'rag_handler': None  # 稍后初始化
        }
    
    def run_continuous_mining(self):
        """持续挖掘模式"""
        self.logger.info("开始7×24小时因子挖掘...")
        
        try:
            # 启动槽位管理器
            self.slot_manager.start()
            
            # 定期任务
            schedule.every(1).hours.do(self._hourly_tasks)
            schedule.every(6).hours.do(self._report_status)
            schedule.every(24).hours.do(self._daily_synthesis)
            
            # 主循环
            while True:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
                
        except KeyboardInterrupt:
            self.logger.info("收到停止信号，正在关闭...")
            self.stop()
        except Exception as e:
            self.logger.error(f"系统错误: {e}")
            self.stop()
    
    def run_batch_mining(self, target_count: int = 100):
        """批量挖掘模式"""
        self.logger.info(f"开始批量挖掘，目标{target_count}个因子...")
        
        # 更新配置
        self.config.config['mining']['target_count'] = target_count
        
        # 重新初始化槽位管理器
        self.slot_manager = SlotManager(
            config=self.config.config,
            shared_resources=self.shared_resources
        )
        
        # 运行挖掘
        self.slot_manager.start()
        
        # 输出结果
        stats = self.slot_manager.get_stats()
        self.logger.info(f"批量挖掘完成: {stats}")
        
        # 合成因子
        self._synthesize_and_report()
    
    def _hourly_tasks(self):
        """每小时执行的任务"""
        self.logger.info("执行每小时任务...")
        
        # 更新RAG索引
        factors = self.shared_resources['factor_pool'].get_all_factors()
        if factors and len(factors) % 10 == 0:  # 每10个新因子更新一次
            self._update_rag_index(factors)
        
        # 保存进度
        self._save_progress()
    
    def _report_status(self):
        """报告系统状态"""
        stats = self.slot_manager.get_stats()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_generated': stats['total_generated'],
            'total_accepted': stats['total_accepted'],
            'accept_rate': stats['accept_rate'],
            'factor_pool_size': len(self.shared_resources['factor_pool'].factors),
            'idea_pool_size': len(self.shared_resources['idea_pool'].ideas)
        }
        
        self.logger.info(f"系统状态报告: {json.dumps(report, indent=2)}")
        
        # 保存报告
        with open(f"reports/status_{datetime.now().strftime('%Y%m%d_%H%M')}.json", 'w') as f:
            json.dump(report, f, indent=2)
    
    def _daily_synthesis(self):
        """每日因子合成"""
        self.logger.info("执行每日因子合成...")
        self._synthesize_and_report()
    
    def _synthesize_and_report(self):
        """合成因子并生成报告"""
        # 获取所有因子
        all_factors = self.shared_resources['factor_pool'].get_all_factors()
        
        if len(all_factors) < 3:
            self.logger.warning("因子数量不足，跳过合成")
            return
        
        # 分离量价因子和基本面因子
        price_factors = [f for f in all_factors if not hasattr(f, 'is_fundamental')]
        fundamental_factors = [f for f in all_factors if hasattr(f, 'is_fundamental')]
        
        # 合成量价因子
        if len(price_factors) >= 3:
            price_result = self.synthesizer.synthesize_price_factors(price_factors)
            self._evaluate_synthesized_factor(price_result, "price")
        
        # 合成基本面因子
        if len(fundamental_factors) >= 3:
            fundamental_result = self.synthesizer.synthesize_fundamental_factors(fundamental_factors)
            self._evaluate_synthesized_factor(fundamental_result, "fundamental")
    
    def _evaluate_synthesized_factor(self, result: Dict[str, Any], factor_type: str):
        """评估合成因子"""
        synthesized_values = result.get('synthesized_values')
        if synthesized_values is None:
            return
        
        # 计算IC
        if factor_type == "price":
            returns = self.test_returns
        else:
            returns = self.test_returns  # 实际应用中可能需要不同的收益率
        
        # 对齐时间
        common_index = synthesized_values.index.intersection(returns.index)
        if len(common_index) < 20:
            return
        
        aligned_values = synthesized_values.loc[common_index]
        aligned_returns = returns.loc[common_index]
        
        # 计算IC
        ic_ts = aligned_values.corrwith(aligned_returns, axis=1)
        ic_mean = ic_ts.mean()
        
        # 计算多空收益
        rank = aligned_values.rank(axis=1)
        n = rank.shape[1]
        q = 0.2
        
        long = rank >= (1 - q) * n
        short = rank <= q * n
        
        pnl = (
            aligned_returns.where(long).mean(axis=1) -
            aligned_returns.where(short).mean(axis=1)
        )
        cumulative_pnl = pnl.cumsum()
        
        # 生成报告
        report = {
            'factor_type': factor_type,
            'synthesis_method': result.get('method'),
            'ic_mean': float(ic_mean),
            'final_pnl': float(cumulative_pnl.iloc[-1]) if len(cumulative_pnl) > 0 else 0,
            'evaluation_date': datetime.now().isoformat()
        }
        
        self.logger.info(f"合成因子评估结果: {json.dumps(report, indent=2)}")
        
        # 保存报告
        filename = f"reports/synthesis_{factor_type}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
    
    def _update_rag_index(self, factors: List):
        """更新RAG索引"""
        if self.shared_resources['rag_handler'] is None:
            self.shared_resources['rag_handler'] = RAGHandler(
                embedding_model=self.config.get("rag.embedding_model"),
                similarity_threshold=self.config.get("rag.similarity_threshold")
            )
        
        self.shared_resources['rag_handler'].add_factors(factors)
    
    def _save_progress(self):
        """保存进度"""
        progress = {
            'factors': [
                {
                    'expr': factor.expr,
                    'ic_mean': factor.ic_mean,
                    'created_time': factor.created_time
                }
                for factor in self.shared_resources['factor_pool'].factors
            ],
            'ideas': self.shared_resources['idea_pool'].ideas,
            'last_updated': datetime.now().isoformat()
        }
        
        with open('progress/progress_backup.json', 'w') as f:
            json.dump(progress, f, indent=2)
    
    def stop(self):
        """停止系统"""
        self.logger.info("停止因子挖掘系统...")
        self.slot_manager.stop()
        
        # 保存最终结果
        self._save_progress()
        
        # 生成最终报告
        self._report_status()
        
        self.logger.info("系统已停止")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM因子自动挖掘系统')
    parser.add_argument('--mode', type=str, default='batch', 
                       choices=['batch', 'continuous'],
                       help='运行模式: batch(批量挖掘) 或 continuous(持续挖掘)')
    parser.add_argument('--target-count', type=int, default=100,
                       help='批量挖掘模式下的目标因子数量')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（可选）')
    
    args = parser.parse_args()
    
    # 创建系统实例
    system = AlphaMiningSystem(config_path=args.config)
    
    try:
        if args.mode == 'batch':
            # 批量挖掘模式
            print(f"开始批量挖掘，目标因子数量: {args.target_count}")
            system.run_batch_mining(target_count=args.target_count)
        else:
            # 持续挖掘模式（7×24小时）
            print("开始7×24小时持续挖掘...")
            system.run_continuous_mining()
    except KeyboardInterrupt:
        print("\n收到停止信号，正在关闭系统...")
        system.stop()
    except Exception as e:
        print(f"系统运行出错: {e}")
        import traceback
        traceback.print_exc()
        system.stop()


if __name__ == "__main__":
    main()