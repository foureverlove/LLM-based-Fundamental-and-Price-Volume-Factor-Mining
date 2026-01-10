# ==================== factor/selector.py ====================
"""
改进的MMR因子选择器
"""
import numpy as np
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass
from .factor import Factor

@dataclass
class CorrelationMetrics:
    """相关性指标"""
    cross_sectional: float  # 截面相关性
    time_series: float     # 时序相关性
    barra_exposure: Dict[str, float]  # Barra因子暴露

class ImprovedMMRSelector:
    """改进的MMR选择器（考虑截面、时序相关性和Barra暴露）"""
    
    def __init__(self, lambda_param: float = 0.7, alpha: float = 0.5,
                 barra_factors: Dict[str, pd.DataFrame] = None):
        """
        Args:
            lambda_param: MMR中的λ参数
            alpha: 截面和时序相关性权重
            barra_factors: Barra风险因子数据
        """
        self.lambda_param = lambda_param
        self.alpha = alpha
        self.barra_factors = barra_factors or {}
        
    def calculate_correlation(self, f1: Factor, f2: Factor, 
                            barra_data: pd.DataFrame = None) -> CorrelationMetrics:
        """计算因子间的相关性指标"""
        
        # 截面相关性
        cs_corr = self._cross_sectional_correlation(f1.values, f2.values)
        
        # 时序相关性
        ts_corr = self._time_series_correlation(f1.ic_ts, f2.ic_ts)
        
        # Barra因子暴露
        barra_exposure = {}
        if self.barra_factors and barra_data is not None:
            barra_exposure = self._calculate_barra_exposure(f1.values, barra_data)
        
        return CorrelationMetrics(
            cross_sectional=cs_corr,
            time_series=ts_corr,
            barra_exposure=barra_exposure
        )
    
    def _cross_sectional_correlation(self, values1: pd.DataFrame, 
                                   values2: pd.DataFrame) -> float:
        """计算截面相关性（平均截面相关性）"""
        common_dates = values1.index.intersection(values2.index)
        if len(common_dates) < 10:
            return 0.0
        
        correlations = []
        for date in common_dates:
            # 获取共同股票
            common_stocks = set(values1.columns) & set(values2.columns)
            if len(common_stocks) < 10:
                continue
            
            stocks = list(common_stocks)
            try:
                a = values1.loc[date, stocks].fillna(0).values
                b = values2.loc[date, stocks].fillna(0).values
                # 避免 std 为 0 导致 divide-by-zero 警告
                if np.allclose(a.std(), 0) or np.allclose(b.std(), 0):
                    continue
                corr = np.corrcoef(a, b)[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
            except:
                continue
        
        return np.mean(correlations) if correlations else 0.0
    
    def _time_series_correlation(self, ic_ts1: pd.Series, 
                               ic_ts2: pd.Series) -> float:
        """计算时序相关性（IC序列相关性）"""
        common_index = ic_ts1.index.intersection(ic_ts2.index)
        if len(common_index) < 20:
            return 0.0
        
        ic1_aligned = ic_ts1.loc[common_index].fillna(0)
        ic2_aligned = ic_ts2.loc[common_index].fillna(0)
        try:
            a = ic1_aligned.values
            b = ic2_aligned.values
            if np.allclose(a.std(), 0) or np.allclose(b.std(), 0):
                return 0.0
            corr = np.corrcoef(a, b)[0, 1]
            return abs(corr) if not np.isnan(corr) else 0.0
        except:
            return 0.0
    
    def _calculate_barra_exposure(self, factor_values: pd.DataFrame,
                                barra_data: pd.DataFrame) -> Dict[str, float]:
        """计算因子在Barra因子上的暴露"""
        exposures = {}
        
        for barra_name, barra_values in self.barra_factors.items():
            # 对齐日期
            common_dates = factor_values.index.intersection(barra_values.index)
            if len(common_dates) < 20:
                continue
            
            # 计算平均暴露
            daily_exposures = []
            for date in common_dates[:50]:  # 最近50天
                common_stocks = set(factor_values.columns) & set(barra_values.columns)
                if len(common_stocks) < 10:
                    continue
                
                stocks = list(common_stocks)
                try:
                    a = factor_values.loc[date, stocks].fillna(0).values
                    b = barra_values.loc[date, stocks].fillna(0).values
                    if np.allclose(a.std(), 0) or np.allclose(b.std(), 0):
                        continue
                    corr = np.corrcoef(a, b)[0, 1]
                    if not np.isnan(corr):
                        daily_exposures.append(abs(corr))
                except:
                    continue
            
            if daily_exposures:
                exposures[barra_name] = np.mean(daily_exposures)
        
        return exposures
    
    def score(self, candidate: Factor, selected: List[Factor], 
             barra_data: pd.DataFrame = None) -> float:
        """
        计算MMR分数
        
        MMR(f_i) = λ·IC(f_i) - (1-λ)·max_{f_j∈S_s∪S_m} (α·Rel_cs(f_i,f_j) + (1-α)·Rel_ts(f_i,f_j))
        """
        if not selected:
            return candidate.ic_mean
        
        max_correlation = 0.0
        candidate_metrics_list = []
        
        for selected_factor in selected:
            metrics = self.calculate_correlation(candidate, selected_factor, barra_data)
            candidate_metrics_list.append(metrics)
            
            # 综合相关性
            combined_corr = (
                self.alpha * metrics.cross_sectional + 
                (1 - self.alpha) * metrics.time_series
            )
            
            # 考虑Barra暴露惩罚
            barra_penalty = 0.0
            for exposure in metrics.barra_exposure.values():
                if exposure > 0.3:  # 暴露过高惩罚
                    barra_penalty += exposure * 0.1
            
            combined_corr += barra_penalty
            max_correlation = max(max_correlation, combined_corr)
        
        # 计算MMR分数
        mmr_score = (
            self.lambda_param * candidate.ic_mean - 
            (1 - self.lambda_param) * max_correlation
        )
        
        return mmr_score