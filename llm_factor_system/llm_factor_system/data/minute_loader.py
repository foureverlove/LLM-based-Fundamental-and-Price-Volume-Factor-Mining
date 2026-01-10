import os
import pandas as pd
from typing import Optional

MINUTE_PATH = "D:/量化/data/minute_data"

class MinuteDataLoader:
    """分钟频数据加载器"""
    
    def __init__(self, base_path: str = None):
        self.base_path = base_path or MINUTE_PATH
        self.cache = {}
    
    def load_day(self, date: str) -> pd.DataFrame:
        """
        加载单日分钟数据
        
        Args:
            date: 日期字符串，格式 '2025-12-16' 或 '20251216'
        
        Returns:
            DataFrame with columns: ['time', 'open', 'high', 'low', 'close', 'volume', 'amount', 'date', 'stock_code']
        """
        # 统一日期格式为 YYYY-MM-DD
        if len(date) == 8:  # 20251216
            date_str = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
        else:
            date_str = date
        
        path = os.path.join(self.base_path, f"{date_str}.parquet")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"分钟数据文件不存在: {path}")
        
        df = pd.read_parquet(path)
        
        # 确保time列为datetime类型
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        
        # 按股票代码和时间排序
        if 'stock_code' in df.columns and 'time' in df.columns:
            df = df.sort_values(['stock_code', 'time'])
        
        return df
    
    def load_range(self, start: str, end: str) -> pd.DataFrame:
        """
        加载日期范围内的分钟数据
        
        Args:
            start: 开始日期 '2025-01-01'
            end: 结束日期 '2025-12-31'
        
        Returns:
            合并后的DataFrame
        """
        dates = pd.date_range(start, end, freq='D')
        dfs = []
        
        for d in dates:
            date_str = d.strftime('%Y-%m-%d')
            try:
                df = self.load_day(date_str)
                dfs.append(df)
            except FileNotFoundError:
                # 跳过不存在的文件
                continue
        
        if not dfs:
            return pd.DataFrame()
        
        return pd.concat(dfs, ignore_index=True)
    
    def load_stock(self, stock_code: str, start: str, end: str) -> pd.DataFrame:
        """
        加载指定股票在日期范围内的分钟数据
        
        Args:
            stock_code: 股票代码
            start: 开始日期
            end: 结束日期
        
        Returns:
            该股票的分钟数据
        """
        df = self.load_range(start, end)
        if 'stock_code' in df.columns:
            return df[df['stock_code'] == stock_code].copy()
        return pd.DataFrame()
