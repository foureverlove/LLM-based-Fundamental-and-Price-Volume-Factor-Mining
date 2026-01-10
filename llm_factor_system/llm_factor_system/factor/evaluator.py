# ==================== factor/evaluator.py ====================
"""
增强版因子评估器
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from scipy import stats

class FactorEvaluator:
    """增强版因子评估器"""
    
    def __init__(self):
        pass
    
    def calc_ic(self, factor_values: pd.DataFrame, 
                returns: pd.DataFrame) -> Tuple[float, pd.Series]:
        """计算IC（信息系数）"""
        # 对齐数据
        common_index = factor_values.index.intersection(returns.index)
        if len(common_index) < 20:
            return 0.0, pd.Series()
        
        factor_aligned = factor_values.loc[common_index]
        returns_aligned = returns.loc[common_index]
        
        # 计算Rank IC（斯皮尔曼相关系数）
        ic_ts = factor_aligned.rank(axis=1).corrwith(
            returns_aligned.rank(axis=1), axis=1
        )
        
        # 去除极端值
        ic_ts = ic_ts.clip(-0.5, 0.5)
        
        ic_mean = ic_ts.mean()
        return ic_mean, ic_ts
    
    def calc_ir(self, ic_ts: pd.Series) -> float:
        """计算IR（信息比率）"""
        if len(ic_ts) < 20:
            return 0.0
        
        ic_mean = ic_ts.mean()
        ic_std = ic_ts.std()
        
        if ic_std == 0:
            return 0.0
        
        ir = ic_mean / ic_std * np.sqrt(252)  # 年化
        return ir
    
    def calc_performance_metrics(self, factor_values: pd.DataFrame,
                               returns: pd.DataFrame,
                               q: float = 0.1) -> Dict[str, Any]:
        """计算完整的绩效指标"""
        
        # 计算IC
        ic_mean, ic_ts = self.calc_ic(factor_values, returns)
        ir = self.calc_ir(ic_ts)
        
        # 计算多空收益
        pnl_series, long_returns, short_returns = self._calc_long_short_returns(
            factor_values, returns, q
        )
        
        # 计算超额收益（相对于基准）
        excess_returns = self._calc_excess_returns(long_returns, returns)
        
        # 计算各项指标
        metrics = {
            'ic_mean': ic_mean,
            'ic_ir': ir,
            'ic_t_stat': self._calc_t_stat(ic_ts),
            'ic_positive_ratio': (ic_ts > 0).mean(),
            'long_annual_return': long_returns.mean() * 252,
            'short_annual_return': short_returns.mean() * 252,
            'ls_annual_return': pnl_series.mean() * 252 if len(pnl_series) > 0 else 0,
            'excess_annual_return': excess_returns.mean() * 252 if len(excess_returns) > 0 else 0,
            'sharpe_ratio': self._calc_sharpe_ratio(pnl_series),
            'max_drawdown': self._calc_max_drawdown(pnl_series.cumsum()),
            'win_rate': (pnl_series > 0).mean() if len(pnl_series) > 0 else 0,
            'profit_factor': self._calc_profit_factor(pnl_series),
            'turnover': self._estimate_turnover(factor_values, q),
            'group_returns': self._calc_group_returns(factor_values, returns, n_groups=10)
        }
        
        return metrics
    
    def _calc_long_short_returns(self, factor_values: pd.DataFrame,
                               returns: pd.DataFrame,
                               q: float = 0.1) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算多空组合收益"""
        # 截面排名
        rank = factor_values.rank(axis=1, pct=True)
        n = rank.shape[1]
        
        # 多头：前q%
        long_mask = rank >= (1 - q)
        # 空头：后q%
        short_mask = rank <= q
        
        # 计算组合收益（等权）
        long_returns = returns.where(long_mask).mean(axis=1)
        short_returns = returns.where(short_mask).mean(axis=1)
        
        # 多空收益
        pnl_series = long_returns - short_returns
        
        return pnl_series, long_returns, short_returns
    
    def _calc_excess_returns(self, strategy_returns: pd.Series,
                           market_returns: pd.DataFrame) -> pd.Series:
        """计算超额收益（相对于全市场等权）"""
        market_ew = market_returns.mean(axis=1)
        
        # 对齐时间
        common_idx = strategy_returns.index.intersection(market_ew.index)
        if len(common_idx) == 0:
            return pd.Series()
        
        strategy_aligned = strategy_returns.loc[common_idx]
        market_aligned = market_ew.loc[common_idx]
        
        return strategy_aligned - market_aligned
    
    def _calc_sharpe_ratio(self, returns: pd.Series) -> float:
        """计算夏普比率"""
        if len(returns) < 20:
            return 0.0
        
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        
        if annual_vol == 0:
            return 0.0
        
        return annual_return / annual_vol
    
    def _calc_max_drawdown(self, cumulative_pnl: pd.Series) -> float:
        """计算最大回撤"""
        if len(cumulative_pnl) < 20:
            return 0.0
        
        running_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - running_max) / (running_max + 1e-10)
        
        return abs(drawdown.min())
    
    def _calc_profit_factor(self, returns: pd.Series) -> float:
        """计算盈亏比"""
        if len(returns) == 0:
            return 0.0
        
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        
        if gross_loss == 0:
            return 10.0  # 无穷大，用大数代替
        
        return gross_profit / gross_loss
    
    def _calc_t_stat(self, series: pd.Series) -> float:
        """计算t统计量"""
        if len(series) < 2:
            return 0.0
        
        mean = series.mean()
        std = series.std()
        
        if std == 0:
            return 0.0
        
        return mean / (std / np.sqrt(len(series)))
    
    def _estimate_turnover(self, factor_values: pd.DataFrame, q: float) -> float:
        """估计换手率"""
        if len(factor_values) < 10:
            return 0.0
        
        # 计算多头组合换手
        rank = factor_values.rank(axis=1, pct=True)
        long_positions = rank >= (1 - q)
        
        # 计算每日持仓变化
        turnover = long_positions.diff().abs().mean().mean()
        
        return turnover * 252  # 年化换手率
    
    def _calc_group_returns(self, factor_values: pd.DataFrame,
                          returns: pd.DataFrame,
                          n_groups: int = 10) -> Dict[str, float]:
        """计算分组收益"""
        group_returns = {}
        
        for i in range(n_groups):
            # 创建分组掩码
            rank = factor_values.rank(axis=1, pct=True)
            lower_bound = i / n_groups
            upper_bound = (i + 1) / n_groups
            
            mask = (rank >= lower_bound) & (rank < upper_bound)
            
            # 计算分组收益
            group_ret = returns.where(mask).mean(axis=1)
            annual_ret = group_ret.mean() * 252 if len(group_ret) > 0 else 0
            
            group_returns[f'group_{i+1}'] = annual_ret
        
        return group_returns
    
    def simple_long_short(self, factor_values: pd.DataFrame,
                        returns: pd.DataFrame,
                        q: float = 0.2) -> pd.Series:
        """简化的多空收益计算（兼容旧版本）"""
        pnl_series, _, _ = self._calc_long_short_returns(factor_values, returns, q)
        return pnl_series.cumsum()