# ==================== factor/factor.py ====================
"""
因子类定义
"""
import pandas as pd
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

@dataclass
class Factor:
    """因子类"""
    expr: str                    # 因子表达式
    explanation: str             # 解释说明
    values: pd.DataFrame         # 因子值
    ic_mean: float              # IC均值
    ic_ts: pd.Series            # IC时间序列
    pnl: pd.Series              # 多空净值曲线
    created_time: str = field(default_factory=lambda: pd.Timestamp.now().isoformat())
    
    # 测试结果
    test_metrics: Optional[Dict[str, Any]] = None
    
    # 相关性信息
    correlations: Dict[str, float] = field(default_factory=dict)
    barra_exposures: Dict[str, float] = field(default_factory=dict)