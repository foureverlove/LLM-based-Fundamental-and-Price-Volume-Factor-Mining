# ==================== operators/fundamental_operators.py ====================
"""
基本面因子专用算子
"""
import pandas as pd
import numpy as np
from typing import Union, List, Dict
from datetime import datetime, timedelta

class FundamentalOperators:
    """基本面因子算子库"""
    
    @staticmethod
    def cs_rank(x: pd.DataFrame) -> pd.DataFrame:
        """截面排名"""
        return x.rank(axis=1, pct=True)
    
    @staticmethod
    def cs_normalize(x: pd.DataFrame) -> pd.DataFrame:
        """截面Min-Max归一化"""
        return x.sub(x.min(axis=1), axis=0).div(
            x.max(axis=1) - x.min(axis=1) + 1e-10, axis=0
        )
    
    @staticmethod
    def cs_zscore(x: pd.DataFrame) -> pd.DataFrame:
        """截面Z-Score标准化"""
        return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1) + 1e-10, axis=0)
    
    @staticmethod
    def yoy_chg(x: pd.DataFrame, n: int = 1) -> pd.DataFrame:
        """
        同比变化率
        (当期值 - N年前同期值) / N年前同期值
        """
        # 假设x是季度数据，索引为季度末日期
        result = pd.DataFrame(index=x.index, columns=x.columns)
        
        for i in range(n * 4, len(x)):  # 4季度=1年
            current = x.iloc[i]
            previous = x.iloc[i - n * 4]
            result.iloc[i] = (current - previous) / (abs(previous) + 1e-10)
        
        return result
    
    @staticmethod
    def yoy_var(x: pd.DataFrame, n: int = 1) -> pd.DataFrame:
        """同比方差"""
        result = pd.DataFrame(index=x.index, columns=x.columns)
        
        for i in range(n * 4, len(x)):
            window = x.iloc[i - n * 4:i:4]  # 取过去n年同期数据
            result.iloc[i] = window.var()
        
        return result
    
    @staticmethod
    def mom_last_zscore(x: pd.DataFrame, n: int = 4) -> pd.DataFrame:
        """
        回顾过去N个季度，计算最后一个值的Z分数
        (最后值 - 均值) / 标准差
        """
        result = pd.DataFrame(index=x.index, columns=x.columns)
        
        for i in range(n, len(x)):
            window = x.iloc[i-n:i]
            last_value = window.iloc[-1]
            mean_value = window.mean()
            std_value = window.std() + 1e-10
            result.iloc[i] = (last_value - mean_value) / std_value
        
        return result
    
    @staticmethod
    def simple_forward_fill(x: pd.DataFrame, n: int = 4) -> pd.DataFrame:
        """
        回顾过去N个季度值，计算变化率
        (当期值 - 前期值) / 前期值
        """
        return x.pct_change(periods=n).fillna(0)
    
    @staticmethod
    def ref(x: pd.DataFrame, n: int) -> pd.DataFrame:
        """N个季度前的值"""
        return x.shift(n)
    
    @staticmethod
    def roll_sum(x: pd.DataFrame, n: int) -> pd.DataFrame:
        """滚动N个季度求和"""
        return x.rolling(n, min_periods=1).sum()
    
    @staticmethod
    def roll_mean(x: pd.DataFrame, n: int) -> pd.DataFrame:
        """滚动N个季度均值"""
        return x.rolling(n, min_periods=1).mean()
    
    @staticmethod
    def roll_std(x: pd.DataFrame, n: int) -> pd.DataFrame:
        """滚动N个季度标准差"""
        return x.rolling(n, min_periods=1).std()
    
    @staticmethod
    def wma(x: pd.DataFrame, n: int) -> pd.DataFrame:
        """加权移动平均"""
        weights = np.arange(1, n + 1)
        weights = weights / weights.sum()
        
        def weighted_mean(series):
            if len(series) < n:
                return np.nan
            return np.dot(series[-n:], weights)
        
        return x.rolling(n, min_periods=n).apply(weighted_mean)