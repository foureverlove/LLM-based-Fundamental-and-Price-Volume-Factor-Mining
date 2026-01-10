import os
import pandas as pd
from typing import Optional, List

DAILY_PATH = "D:/量化/data/daily_data"

class DailyDataLoader:
    """日频数据加载器"""
    
    def __init__(self, base_path: str = None):
        self.base_path = base_path or DAILY_PATH
        self.cache = {}
        
        # 字段名映射（大写转小写）
        self.field_mapping = {
            'Close': 'close',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Volume': 'volume',
            'Amount': 'amount',
            'Vwap': 'vwap',
            'Turnover': 'turnover',
            'FloatMarketSize': 'float_market_size',
            'OutstandingShare': 'outstanding_share'
        }
    
    def load_field(self, field: str) -> pd.DataFrame:
        """
        加载指定字段的日频数据
        
        Args:
            field: 字段名，如 'Close', 'close', 'Amount', 'amount' 等
        
        Returns:
            DataFrame: columns为stock_code，index为日期
        """
        # 统一转换为小写字段名
        field_lower = field.lower()
        if field in self.field_mapping:
            field_lower = self.field_mapping[field].lower()
        
        # 检查缓存
        cache_key = field_lower
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 构建文件路径
        file_name = f"{field_lower}.parquet"
        path = os.path.join(self.base_path, file_name)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"日频数据文件不存在: {path}")
        
        df = pd.read_parquet(path)
        
        # 确保index为datetime类型
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # 确保columns为stock_code（如果需要转换）
        # 假设数据已经是正确的格式：columns为stock_code，index为日期
        
        # 缓存数据
        self.cache[cache_key] = df
        
        return df
    
    def load_all_fields(self) -> dict:
        """
        加载所有可用字段
        
        Returns:
            dict: {字段名: DataFrame}
        """
        available_fields = [
            'close', 'open', 'high', 'low', 'volume', 'amount', 'vwap',
            'turnover', 'float_market_size', 'outstanding_share'
        ]
        
        result = {}
        for field in available_fields:
            try:
                result[field] = self.load_field(field)
            except FileNotFoundError:
                continue
        
        return result
    
    def get_available_fields(self) -> List[str]:
        """
        获取可用的字段列表
        
        Returns:
            可用字段名列表
        """
        available_fields = []
        field_names = [
            'close', 'open', 'high', 'low', 'volume', 'amount', 'vwap',
            'turnover', 'float_market_size', 'outstanding_share'
        ]
        
        for field in field_names:
            file_path = os.path.join(self.base_path, f"{field}.parquet")
            if os.path.exists(file_path):
                available_fields.append(field)
        
        return available_fields
