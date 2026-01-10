# ==================== factor/factor_pool.py ====================
"""
因子池管理器
"""
import threading
import json
from typing import List, Dict, Any
from datetime import datetime
from .factor import Factor
from .selector import ImprovedMMRSelector

class FactorPool:
    """因子池管理器"""
    
    def __init__(self, max_size: int = 1000):
        self.factors = []  # 存储因子对象
        self.lock = threading.Lock()
        self.max_size = max_size
        self.selector = ImprovedMMRSelector()
        
    def add_factor(self, factor: Factor) -> bool:
        """添加因子到池中"""
        with self.lock:
            # 检查是否已存在类似因子
            if self._is_duplicate(factor):
                return False
            
            # 检查池是否已满
            if len(self.factors) >= self.max_size:
                # 移除表现最差的因子
                self.factors.sort(key=lambda x: abs(x.ic_mean))
                self.factors = self.factors[:self.max_size - 1]
            
            self.factors.append(factor)
            return True
    
    def _is_duplicate(self, new_factor: Factor, threshold: float = 0.9) -> bool:
        """检查是否为重复因子"""
        for existing in self.factors:
            # 计算表达式相似度
            similarity = self._expression_similarity(new_factor.expr, existing.expr)
            if similarity > threshold:
                return True
            
            # 计算IC序列相关性
            if len(new_factor.ic_ts) > 10 and len(existing.ic_ts) > 10:
                # 对齐时间
                common_idx = new_factor.ic_ts.index.intersection(existing.ic_ts.index)
                if len(common_idx) > 10:
                    corr = new_factor.ic_ts.loc[common_idx].corr(existing.ic_ts.loc[common_idx])
                    if abs(corr) > threshold:
                        return True
        return False
    
    def _expression_similarity(self, expr1: str, expr2: str) -> float:
        """计算表达式相似度"""
        # 简单实现：基于字符串编辑距离
        from difflib import SequenceMatcher
        return SequenceMatcher(None, expr1, expr2).ratio()
    
    def get_all_factors(self) -> List[Factor]:
        """获取所有因子"""
        with self.lock:
            return self.factors.copy()
    
    def get_best_factors(self, n: int = 10, 
                        min_ic: float = 0.02,
                        min_excess_return: float = 0.05) -> List[Factor]:
        """获取表现最好的因子"""
        with self.lock:
            filtered = [
                f for f in self.factors 
                if abs(f.ic_mean) >= min_ic
            ]
            
            # 按IC绝对值排序
            filtered.sort(key=lambda x: abs(x.ic_mean), reverse=True)
            return filtered[:n]
    
    def get_low_correlation_factors(self, target_factor: Factor,
                                  max_corr: float = 0.7,
                                  n: int = 5) -> List[Factor]:
        """获取与目标因子相关性低的因子"""
        with self.lock:
            low_corr_factors = []
            for factor in self.factors:
                if factor == target_factor:
                    continue
                
                # 计算相关性
                correlation = self.selector.calculate_correlation(
                    target_factor, factor
                )
                total_corr = (self.selector.alpha * correlation.cross_sectional + 
                            (1 - self.selector.alpha) * correlation.time_series)
                
                if abs(total_corr) <= max_corr:
                    low_corr_factors.append((factor, total_corr))
            
            # 按相关性排序（从低到高）
            low_corr_factors.sort(key=lambda x: abs(x[1]))
            return [f[0] for f in low_corr_factors[:n]]
    
    def save(self, path: str):
        """保存因子池"""
        with self.lock:
            data = []
            for factor in self.factors:
                factor_data = {
                    'expr': factor.expr,
                    'explanation': factor.explanation,
                    'ic_mean': factor.ic_mean,
                    'created_time': factor.created_time,
                    'test_metrics': factor.test_metrics,
                    'correlations': factor.correlations,
                    'barra_exposures': factor.barra_exposures
                }
                data.append(factor_data)
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
    
    def load(self, path: str):
        """加载因子池"""
        with self.lock:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # 注意：需要重新计算因子值，这里只加载元数据
            self.factors = []
            for factor_data in data:
                factor = Factor(
                    expr=factor_data['expr'],
                    explanation=factor_data['explanation'],
                    values=None,  # 需要重新计算
                    ic_mean=factor_data['ic_mean'],
                    ic_ts=None,
                    pnl=None
                )
                factor.created_time = factor_data['created_time']
                factor.test_metrics = factor_data.get('test_metrics')
                factor.correlations = factor_data.get('correlations', {})
                factor.barra_exposures = factor_data.get('barra_exposures', {})
                self.factors.append(factor)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            if not self.factors:
                return {}
            
            ics = [f.ic_mean for f in self.factors]
            return {
                'total_factors': len(self.factors),
                'avg_ic': sum(ics) / len(ics),
                'max_ic': max(ics, key=abs),
                'positive_ratio': sum(1 for ic in ics if ic > 0) / len(ics),
                'recent_additions': len([f for f in self.factors 
                                       if datetime.now().timestamp() - 
                                       datetime.fromisoformat(f.created_time).timestamp() 
                                       < 24*3600])
            }