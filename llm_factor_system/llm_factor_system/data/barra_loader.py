# ==================== data/barra_loader.py ====================
"""
Barra风险因子加载器
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import os

class BarraLoader:
    """Barra风险因子加载器"""
    
    def __init__(self, base_path: str = None):
        """
        Args:
            base_path: 数据路径，默认为日频数据路径
        """
        if base_path is None:
            # 默认使用日频数据路径（Barra因子和日频数据在同一目录）
            from .daily_loader import DAILY_PATH
            self.base_path = DAILY_PATH
        else:
            self.base_path = base_path
        
        self.factors = {}
        self.cache = {}
        
        # Barra因子文件映射（因子名 -> 文件名）
        self.factor_file_mapping = {
            'LNSIZE': 'lnsize.parquet',
            'LIQUIDITY': 'liquidity.parquet',
            'MIDCAP': 'MidCap.parquet',
            'MOMENTUM': 'MOMENTUM.parquet',
            'VALUE': 'BarraValue.parquet',
            'VOLATILITY': 'Volatility.parquet',
            'DIVIDEND': 'DividendYield.parquet'
        }
    
    def load_factor(self, factor_name: str) -> Optional[pd.DataFrame]:
        """
        加载单个Barra因子
        
        Args:
            factor_name: 因子名称，如 'LNSIZE', 'LIQUIDITY' 等
        
        Returns:
            DataFrame: columns为stock_code，index为日期
        """
        factor_name_upper = factor_name.upper()
        
        # 检查缓存
        if factor_name_upper in self.cache:
            return self.cache[factor_name_upper]
        
        # 获取文件名
        file_name = self.factor_file_mapping.get(factor_name_upper)
        if file_name is None:
            # 尝试直接使用因子名作为文件名
            file_name = f"{factor_name.lower()}.parquet"
        
        path = os.path.join(self.base_path, file_name)
        
        if not os.path.exists(path):
            print(f"警告: Barra因子文件不存在: {path}")
            return None
        
        df = pd.read_parquet(path)
        
        # 确保index为datetime类型
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # 缓存
        self.cache[factor_name_upper] = df
        self.factors[factor_name_upper] = df
        
        return df
    
    def load_all_factors(self) -> Dict[str, pd.DataFrame]:
        """
        加载所有可用的Barra因子
        
        Returns:
            dict: {因子名: DataFrame}
        """
        loaded_factors = {}
        
        for factor_name in self.factor_file_mapping.keys():
            df = self.load_factor(factor_name)
            if df is not None:
                loaded_factors[factor_name] = df
        
        return loaded_factors
    
    def get_available_factors(self) -> List[str]:
        """
        获取可用的因子列表
        
        Returns:
            可用因子名列表
        """
        available = []
        for factor_name, file_name in self.factor_file_mapping.items():
            if file_name is None:
                continue
            path = os.path.join(self.base_path, file_name)
            if os.path.exists(path):
                available.append(factor_name)
        return available
    
    def get_factor_exposure(self, factor_values: pd.DataFrame, 
                          date: pd.Timestamp) -> Dict[str, float]:
        """计算因子在Barra因子上的暴露"""
        exposures = {}
        
        for factor_name, factor_data in self.factors.items():
            if date in factor_data.index:
                # 计算相关性作为暴露
                common_stocks = set(factor_values.columns) & set(factor_data.loc[date].index)
                if len(common_stocks) > 10:  # 至少10只股票
                    stocks_list = list(common_stocks)
                    a = factor_values.loc[date, stocks_list].values
                    b = factor_data.loc[date, stocks_list].values
                    if np.allclose(a.std(), 0) or np.allclose(b.std(), 0):
                        exposures[factor_name] = 0
                    else:
                        corr = np.corrcoef(a, b)[0, 1]
                        exposures[factor_name] = corr if not np.isnan(corr) else 0
        
        return exposures