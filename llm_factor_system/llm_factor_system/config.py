"""
配置管理文件
"""
import os
from datetime import datetime
from typing import Dict, Any
import yaml

class Config:
    """配置管理器"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_default_config()
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                self.config.update(user_config)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        return {
            # 路径配置
            "data_paths": {
                "minute": "D:/量化/data/minute_data/",
                "daily": "D:/量化/data/daily_data/",
                "barra": "D:/量化/data/daily_data/"  # Barra因子和日频数据在同一目录
            },
            
            # 时间范围配置
            "time_ranges": {
                "train_start": "2022-01-01",
                "train_end": "2024-12-31",
                "test_start": "2025-01-01",
                "test_end": "2025-12-31"
            },
            
            # 因子挖掘配置
            "mining": {
                "max_parallel_slots": 8,
                "batch_size": 10,
                "ic_threshold": 0.02,
                "excess_return_threshold": 0.05,
                "max_correlation": 0.7,
                "mmr_lambda": 0.7,
                "retry_times": 3
            },
            
            # LLM配置
            "llm": {
                "model": "gpt-4.1-mini",
                "temperature": 0.1,
                "max_tokens": 2000,
                "timeout": 30,
                "api_key": 'sk-sRF9BzJB5viwprPh0PqkeH6Epo01ssFl4msRJl8rpB72FbdC',  # 从环境变量读取
                "api_base": 'https://api.openai-proxy.org/v1'  # 从环境变量读取
            },
            
            # RAG配置
            "rag": {
                "top_k": 3,
                "similarity_threshold": 0.8,
                "embedding_model": "text-embedding-ada-002"
            },
            
            # 日志配置
            "logging": {
                "level": "INFO",
                "file": f"logs/alpha_mining_{datetime.now().strftime('%Y%m%d')}.log",
                "max_size": 100 * 1024 * 1024,  # 100MB
                "backup_count": 10
            }
        }
    
    def get(self, key: str, default=None):
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k)
            if value is None:
                return default
        return value