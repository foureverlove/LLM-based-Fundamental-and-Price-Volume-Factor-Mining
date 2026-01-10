# ==================== operators/price_operators.py ====================
"""
量价因子专用算子
基于研报中的量价因子设计，包含各种技术指标和价格模式算子
"""
import pandas as pd
import numpy as np
from typing import Union, Dict, Callable, Any
from scipy import stats

class PriceOperators:
    """量价因子算子库"""
    
    @staticmethod
    def slope(x: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        计算斜率
        Args:
            x: 价格序列
            window: 窗口大小
        Returns:
            斜率值
        """
        def calc_slope(series):
            if len(series) < 2:
                return np.nan
            try:
                return np.polyfit(range(len(series)), series, 1)[0]
            except:
                return np.nan
        
        return x.rolling(window, min_periods=int(window/2)).apply(calc_slope, raw=True)
    
    @staticmethod
    def mean(x: pd.DataFrame, window: int) -> pd.DataFrame:
        """移动平均值"""
        return x.rolling(window, min_periods=int(window/2)).mean()
    
    @staticmethod
    def std(x: pd.DataFrame, window: int) -> pd.DataFrame:
        """移动标准差"""
        return x.rolling(window, min_periods=int(window/2)).std()
    
    @staticmethod
    def ema(x: pd.DataFrame, window: int) -> pd.DataFrame:
        """指数移动平均"""
        return x.ewm(span=window, min_periods=int(window/2)).mean()
    
    @staticmethod
    def sma(x: pd.DataFrame, window: int) -> pd.DataFrame:
        """简单移动平均"""
        return x.rolling(window, min_periods=int(window/2)).mean()
    
    @staticmethod
    def wma(x: pd.DataFrame, window: int) -> pd.DataFrame:
        """加权移动平均"""
        weights = np.arange(1, window + 1)
        weights = weights / weights.sum()
        
        def weighted_mean(series):
            if len(series) < window:
                return np.nan
            return np.dot(series[-window:], weights)
        
        return x.rolling(window, min_periods=window).apply(weighted_mean, raw=True)
    
    @staticmethod
    def max(x: pd.DataFrame, window: int) -> pd.DataFrame:
        """滚动最大值"""
        return x.rolling(window, min_periods=int(window/2)).max()
    
    @staticmethod
    def min(x: pd.DataFrame, window: int) -> pd.DataFrame:
        """滚动最小值"""
        return x.rolling(window, min_periods=int(window/2)).min()
    
    @staticmethod
    def median(x: pd.DataFrame, window: int) -> pd.DataFrame:
        """滚动中位数"""
        return x.rolling(window, min_periods=int(window/2)).median()
    
    @staticmethod
    def ref(x: pd.DataFrame, n: int) -> pd.DataFrame:
        """滞后n期"""
        return x.shift(n)
    
    @staticmethod
    def delta(x: pd.DataFrame, n: int) -> pd.DataFrame:
        """n期变化量"""
        return x - x.shift(n)
    
    @staticmethod
    def pct_change(x: pd.DataFrame, n: int) -> pd.DataFrame:
        """n期变化率"""
        return x.pct_change(n)
    
    @staticmethod
    def corr(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        滚动相关性
        Args:
            x: 第一个序列
            y: 第二个序列
            window: 窗口大小
        Returns:
            相关系数
        """
        def calc_corr(x_series, y_series):
            if len(x_series) < 2 or len(y_series) < 2:
                return np.nan
            try:
                a = np.array(x_series)
                b = np.array(y_series)
                if np.allclose(a.std(), 0) or np.allclose(b.std(), 0):
                    return np.nan
                return np.corrcoef(a, b)[0, 1]
            except:
                return np.nan
        
        # 确保形状一致
        result = pd.DataFrame(index=x.index, columns=x.columns)
        
        for col in x.columns:
            if col in y.columns:
                # 对齐两个序列
                x_col = x[col]
                y_col = y[col]
                
                # 计算滚动相关性
                corr_values = []
                for i in range(len(x_col)):
                    if i < window - 1:
                        corr_values.append(np.nan)
                    else:
                        x_window = x_col.iloc[i-window+1:i+1]
                        y_window = y_col.iloc[i-window+1:i+1]
                        corr = calc_corr(x_window.values, y_window.values)
                        corr_values.append(corr)
                
                result[col] = corr_values
        
        return result
    
    @staticmethod
    def cov(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
        """滚动协方差"""
        return x.rolling(window).cov(y)
    
    @staticmethod
    def var(x: pd.DataFrame, window: int) -> pd.DataFrame:
        """滚动方差"""
        return x.rolling(window, min_periods=int(window/2)).var()
    
    @staticmethod
    def zscore(x: pd.DataFrame, window: int) -> pd.DataFrame:
        """滚动Z-Score"""
        mean = x.rolling(window, min_periods=int(window/2)).mean()
        std = x.rolling(window, min_periods=int(window/2)).std()
        return (x - mean) / (std + 1e-10)
    
    @staticmethod
    def rank(x: pd.DataFrame) -> pd.DataFrame:
        """截面排名"""
        return x.rank(axis=1, pct=True)
    
    @staticmethod
    def cs_zscore(x: pd.DataFrame) -> pd.DataFrame:
        """截面Z-Score标准化"""
        mean = x.mean(axis=1)
        std = x.std(axis=1)
        return x.sub(mean, axis=0).div(std + 1e-10, axis=0)
    
    @staticmethod
    def cs_normalize(x: pd.DataFrame) -> pd.DataFrame:
        """截面Min-Max归一化"""
        min_val = x.min(axis=1)
        max_val = x.max(axis=1)
        return x.sub(min_val, axis=0).div(max_val - min_val + 1e-10, axis=0)
    
    @staticmethod
    def log(x: pd.DataFrame) -> pd.DataFrame:
        """对数变换"""
        return np.log(np.abs(x) + 1e-10) * np.sign(x)
    
    @staticmethod
    def sign(x: pd.DataFrame) -> pd.DataFrame:
        """符号函数"""
        return np.sign(x)
    
    @staticmethod
    def abs(x: pd.DataFrame) -> pd.DataFrame:
        """绝对值"""
        return np.abs(x)
    
    @staticmethod
    def power(x: pd.DataFrame, n: float) -> pd.DataFrame:
        """幂运算"""
        return np.power(x, n)
    
    @staticmethod
    def sqrt(x: pd.DataFrame) -> pd.DataFrame:
        """平方根"""
        return np.sqrt(np.abs(x)) * np.sign(x)
    
    @staticmethod
    def ts_rank(x: pd.DataFrame, window: int) -> pd.DataFrame:
        """时间序列排名"""
        return x.rolling(window).apply(
            lambda s: stats.percentileofscore(s, s[-1])/100.0 if len(s) == window else np.nan,
            raw=True
        )
    
    @staticmethod
    def ts_min(x: pd.DataFrame, window: int) -> pd.DataFrame:
        """时间序列最小值"""
        return x.rolling(window).min()
    
    @staticmethod
    def ts_max(x: pd.DataFrame, window: int) -> pd.DataFrame:
        """时间序列最大值"""
        return x.rolling(window).max()
    
    @staticmethod
    def ts_argmax(x: pd.DataFrame, window: int) -> pd.DataFrame:
        """最大值位置"""
        def argmax_func(s):
            if len(s) == window:
                return np.argmax(s) / (window - 1)
            return np.nan
        
        return x.rolling(window).apply(argmax_func, raw=True)
    
    @staticmethod
    def ts_argmin(x: pd.DataFrame, window: int) -> pd.DataFrame:
        """最小值位置"""
        def argmin_func(s):
            if len(s) == window:
                return np.argmin(s) / (window - 1)
            return np.nan
        
        return x.rolling(window).apply(argmin_func, raw=True)
    
    @staticmethod
    def decay_linear(x: pd.DataFrame, window: int) -> pd.DataFrame:
        """线性衰减"""
        weights = np.arange(1, window + 1)
        weights = weights / weights.sum()
        
        def linear_decay(series):
            if len(series) < window:
                return np.nan
            return np.dot(series[-window:], weights)
        
        return x.rolling(window, min_periods=window).apply(linear_decay, raw=True)
    
    @staticmethod
    def delay(x: pd.DataFrame, n: int) -> pd.DataFrame:
        """延迟算子"""
        return x.shift(n)
    
    @staticmethod
    def scale(x: pd.DataFrame, scale: float = 1.0) -> pd.DataFrame:
        """缩放"""
        return x / np.abs(x).sum(axis=1).values[:, None] * scale
    
    @staticmethod
    def adv(x: pd.DataFrame, window: int) -> pd.DataFrame:
        """平均成交量"""
        return x.rolling(window, min_periods=int(window/2)).mean()
    
    @staticmethod
    def vwap(x: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        """成交量加权平均价"""
        # 假设x是价格，volume是成交量
        return (x * volume) / (volume + 1e-10)
    
    @staticmethod
    def hl_ratio(x_high: pd.DataFrame, x_low: pd.DataFrame, window: int) -> pd.DataFrame:
        """高低价比率"""
        high_max = x_high.rolling(window, min_periods=int(window/2)).max()
        low_min = x_low.rolling(window, min_periods=int(window/2)).min()
        return (high_max - low_min) / (low_min + 1e-10)
    
    @staticmethod
    def volatility(x: pd.DataFrame, window: int) -> pd.DataFrame:
        """波动率"""
        return x.pct_change().rolling(window, min_periods=int(window/2)).std() * np.sqrt(252)
    
    @staticmethod
    def momentum(x: pd.DataFrame, window: int) -> pd.DataFrame:
        """动量"""
        return x / x.shift(window) - 1
    
    @staticmethod
    def rsi(x: pd.DataFrame, window: int) -> pd.DataFrame:
        """相对强弱指数"""
        delta = x.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(x: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.DataFrame]:
        """MACD指标"""
        ema_fast = x.ewm(span=fast).mean()
        ema_slow = x.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(x: pd.DataFrame, window: int = 20, num_std: float = 2) -> Dict[str, pd.DataFrame]:
        """布林带"""
        sma = x.rolling(window).mean()
        std = x.rolling(window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        
        return {
            'middle': sma,
            'upper': upper,
            'lower': lower
        }
    
    @staticmethod
    def atr(x_high: pd.DataFrame, x_low: pd.DataFrame, x_close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """平均真实波幅"""
        high_low = x_high - x_low
        high_close = np.abs(x_high - x_close.shift())
        low_close = np.abs(x_low - x_close.shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window).mean()
    
    @staticmethod
    def obv(x_close: pd.DataFrame, x_volume: pd.DataFrame) -> pd.DataFrame:
        """能量潮"""
        returns = x_close.pct_change()
        obv = (x_volume * np.sign(returns)).cumsum()
        return obv
    
    @staticmethod
    def volume_ratio(x_volume: pd.DataFrame, window: int) -> pd.DataFrame:
        """成交量比率"""
        return x_volume / x_volume.rolling(window).mean()
    
    @staticmethod
    def money_flow(x_close: pd.DataFrame, x_high: pd.DataFrame, 
                   x_low: pd.DataFrame, x_volume: pd.DataFrame) -> pd.DataFrame:
        """资金流量"""
        typical_price = (x_high + x_low + x_close) / 3
        money_flow = typical_price * x_volume
        return money_flow
    
    @staticmethod
    def price_volume_trend(x_close: pd.DataFrame, x_volume: pd.DataFrame) -> pd.DataFrame:
        """价量趋势"""
        price_change = x_close.pct_change()
        pvt = (price_change * x_volume).cumsum()
        return pvt
    
    @staticmethod
    def elder_ray(x_close: pd.DataFrame, x_high: pd.DataFrame, 
                  x_low: pd.DataFrame, window: int = 13) -> Dict[str, pd.DataFrame]:
        """Elder-Ray指标"""
        ema = x_close.ewm(span=window).mean()
        bull_power = x_high - ema
        bear_power = x_low - ema
        
        return {
            'bull_power': bull_power,
            'bear_power': bear_power,
            'ema': ema
        }
    
    @staticmethod
    def price_position(x: pd.DataFrame, window: int) -> pd.DataFrame:
        """价格位置（在近期区间中的位置）"""
        min_price = x.rolling(window).min()
        max_price = x.rolling(window).max()
        return (x - min_price) / (max_price - min_price + 1e-10)
    
    @staticmethod
    def efficiency_ratio(x: pd.DataFrame, window: int) -> pd.DataFrame:
        """效率比率"""
        price_change = x.diff(window).abs()
        volatility = x.diff().abs().rolling(window).sum()
        return price_change / (volatility + 1e-10)
    
    @staticmethod
    def kurtosis(x: pd.DataFrame, window: int) -> pd.DataFrame:
        """峰度"""
        return x.rolling(window).kurt()
    
    @staticmethod
    def skew(x: pd.DataFrame, window: int) -> pd.DataFrame:
        """偏度"""
        return x.rolling(window).skew()
    
    @staticmethod
    def entropy(x: pd.DataFrame, window: int) -> pd.DataFrame:
        """信息熵"""
        def calc_entropy(series):
            if len(series) < 2:
                return np.nan
            # 归一化
            p = series / (series.sum() + 1e-10)
            # 计算熵
            return -np.sum(p * np.log(p + 1e-10))
        
        return x.rolling(window).apply(calc_entropy, raw=True)
    
    @staticmethod
    def complex_operator(operator_func: Callable, *args, **kwargs) -> Any:
        """
        复杂算子包装器
        允许用户自定义复杂算子
        """
        return operator_func(*args, **kwargs)

# 创建算子库字典，方便调用
PRICE_OPERATOR_LIB = {
    'Slope': PriceOperators.slope,
    'Mean': PriceOperators.mean,
    'Std': PriceOperators.std,
    'EMA': PriceOperators.ema,
    'SMA': PriceOperators.sma,
    'WMA': PriceOperators.wma,
    'Max': PriceOperators.max,
    'Min': PriceOperators.min,
    'Median': PriceOperators.median,
    'Ref': PriceOperators.ref,
    'Delta': PriceOperators.delta,
    'PctChange': PriceOperators.pct_change,
    'Corr': PriceOperators.corr,
    'Cov': PriceOperators.cov,
    'Var': PriceOperators.var,
    'ZScore': PriceOperators.zscore,
    'Rank': PriceOperators.rank,
    'CSZScore': PriceOperators.cs_zscore,
    'CSNormalize': PriceOperators.cs_normalize,
    'Log': PriceOperators.log,
    'Sign': PriceOperators.sign,
    'Abs': PriceOperators.abs,
    'Power': PriceOperators.power,
    'Sqrt': PriceOperators.sqrt,
    'TSRank': PriceOperators.ts_rank,
    'TSMin': PriceOperators.ts_min,
    'TSMax': PriceOperators.ts_max,
    'TSArgmax': PriceOperators.ts_argmax,
    'TSArgmin': PriceOperators.ts_argmin,
    'DecayLinear': PriceOperators.decay_linear,
    'Delay': PriceOperators.delay,
    'Scale': PriceOperators.scale,
    'Adv': PriceOperators.adv,
    'VWAP': PriceOperators.vwap,
    'HLRatio': PriceOperators.hl_ratio,
    'Volatility': PriceOperators.volatility,
    'Momentum': PriceOperators.momentum,
    'RSI': PriceOperators.rsi,
    'VolumeRatio': PriceOperators.volume_ratio,
    'MoneyFlow': PriceOperators.money_flow,
    'PriceVolumeTrend': PriceOperators.price_volume_trend,
    'PricePosition': PriceOperators.price_position,
    'EfficiencyRatio': PriceOperators.efficiency_ratio,
    'Kurtosis': PriceOperators.kurtosis,
    'Skew': PriceOperators.skew,
    'Entropy': PriceOperators.entropy
}

# 研报中示例因子的实现
class ReportExamples:
    """研报中示例因子的实现"""
    
    @staticmethod
    def factor_1(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        """
        研报中的因子1：
        EMA(Slope(close, 5) * Slope(volume, 5), 5)
        """
        close_slope = PriceOperators.slope(close, 5)
        volume_slope = PriceOperators.slope(volume, 5)
        combined = close_slope * volume_slope
        return PriceOperators.ema(combined, 5)
    
    @staticmethod
    def factor_2(close: pd.DataFrame, high: pd.DataFrame, 
                 low: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
        """
        研报中的因子2：
        (close - Max(high, 5)) / (Max(high, 5) - Min(low, 5)) * EMA(volume, 5)
        """
        max_high = PriceOperators.max(high, 5)
        min_low = PriceOperators.min(low, 5)
        price_position = (close - max_high) / (max_high - min_low + 1e-10)
        volume_ema = PriceOperators.ema(volume, 5)
        return price_position * volume_ema
    
    @staticmethod
    def factor_3(close: pd.DataFrame, vwap: pd.DataFrame, 
                 volume: pd.DataFrame, high: pd.DataFrame, 
                 low: pd.DataFrame) -> pd.DataFrame:
        """
        研报中的因子3：
        Mean(volume, 20) * (Max(high, 5) - Min(low, 5)) / (Corr(close, vwap, 10) + 2)
        """
        mean_volume = PriceOperators.mean(volume, 20)
        high_low_range = PriceOperators.max(high, 5) - PriceOperators.min(low, 5)
        corr_close_vwap = PriceOperators.corr(close, vwap, 10)
        denominator = corr_close_vwap + 2
        return mean_volume * high_low_range / (denominator + 1e-10)