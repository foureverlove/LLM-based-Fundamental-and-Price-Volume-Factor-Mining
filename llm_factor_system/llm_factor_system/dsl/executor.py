# ==================== dsl/executor.py ====================
"""
因子表达式执行器
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Union
import re
from datetime import datetime, timedelta
from operators.price_operators import PRICE_OPERATOR_LIB
from operators.fundamental_operators import FundamentalOperators

class DSLExecutor:
    """因子表达式执行器"""
    
    def __init__(self, operator_lib: Dict = None):
        # 合并价格算子和基本面算子
        default_operators = {}
        default_operators.update(PRICE_OPERATOR_LIB)
        
        # 添加基本面算子
        fundamental_ops = {
            'YoYChg': FundamentalOperators.yoy_chg,
            'YoYVar': FundamentalOperators.yoy_var,
            'MomLastZScore': FundamentalOperators.mom_last_zscore,
            'SimpleForwardFill': FundamentalOperators.simple_forward_fill,
            'CSRank': FundamentalOperators.cs_rank,
            'CSNormalize': FundamentalOperators.cs_normalize,
            'CSZScore': FundamentalOperators.cs_zscore,
        }
        default_operators.update(fundamental_ops)
        
        self.operators = operator_lib or default_operators
        
        # 字段别名映射（统一映射到实际数据中的字段名）
        self.field_aliases = {
            'close': 'Close', '收盘价': 'Close', 'Close': 'Close',
            'open': 'Open', '开盘价': 'Open', 'Open': 'Open',
            'high': 'High', '最高价': 'High', 'High': 'High',
            'low': 'Low', '最低价': 'Low', 'Low': 'Low',
            'volume': 'Volume', '成交量': 'Volume', 'Volume': 'Volume',
            'amount': 'Amount', '成交额': 'Amount', 'Amount': 'Amount',
            'vwap': 'Vwap', '均价': 'Vwap', 'Vwap': 'Vwap',
            'returns': 'Returns', '收益率': 'Returns',
            'turnover': 'Turnover', '换手率': 'Turnover',
            'float_market_size': 'FloatMarketSize', '流通市值': 'FloatMarketSize',
            'outstanding_share': 'OutstandingShare', '流通股本': 'OutstandingShare'
        }
    
    def _get_default_operators(self) -> Dict:
        """获取默认算子库"""
        return {
            # 基本运算
            'Add': lambda a, b: a + b,
            'Sub': lambda a, b: a - b,
            'Mul': lambda a, b: a * b,
            'Div': lambda a, b: a / (b + 1e-10),
            
            # 统计算子
            'Mean': lambda x, window: x.rolling(window, min_periods=int(window/2)).mean(),
            'Std': lambda x, window: x.rolling(window, min_periods=int(window/2)).std(),
            'Max': lambda x, window: x.rolling(window, min_periods=int(window/2)).max(),
            'Min': lambda x, window: x.rolling(window, min_periods=int(window/2)).min(),
            'Median': lambda x, window: x.rolling(window, min_periods=int(window/2)).median(),
            
            # 时间序列算子
            'Ref': lambda x, n: x.shift(n),
            'Delta': lambda x, n: x - x.shift(n),
            'Slope': lambda x, window: x.rolling(window).apply(
                lambda s: np.polyfit(range(len(s)), s, 1)[0] if len(s) >= 2 else np.nan
            ),
            'EMA': lambda x, window: x.ewm(span=window, min_periods=int(window/2)).mean(),
            
            # 截面算子
            'Rank': lambda x: x.rank(axis=1, pct=True),
            'ZScore': lambda x: x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1) + 1e-10, axis=0),
            
            # 相关性算子
            'Corr': lambda x, y, window: x.rolling(window).corr(y),
            'Cov': lambda x, y, window: x.rolling(window).cov(y),
            
            # 高级算子
            'Log': lambda x: np.log(abs(x) + 1e-10) * np.sign(x),
            'Sign': lambda x: np.sign(x),
            'Abs': lambda x: np.abs(x),
            'Power': lambda x, n: np.power(x, n)
        }
    
    def run(self, expr: str, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """执行因子表达式"""
        try:
            # 预处理表达式
            expr = self._preprocess_expression(expr)
            
            # 解析并执行
            result = self._eval_expression(expr, data)
            
            # 后处理：去除NaN，截面标准化
            # 使用 ffill() 避免 FutureWarning（fillna(method=...) 已弃用）
            result = result.ffill().fillna(0)
            
            # 截面标准化（可选）
            # result = result.apply(lambda x: (x - x.mean()) / (x.std() + 1e-10), axis=1)
            
            return result
            
        except Exception as e:
            print(f"表达式执行失败: {expr}, 错误: {e}")
            return pd.DataFrame()
    
    def _preprocess_expression(self, expr: str) -> str:
        """预处理表达式"""
        # 移除多余空格
        expr = re.sub(r'\s+', ' ', expr).strip()
        
        # 标准化函数名
        func_mapping = {
            'mean': 'Mean', 'std': 'Std', 'max': 'Max', 'min': 'Min',
            'ema': 'EMA', 'log': 'Log', 'abs': 'Abs', 'sign': 'Sign',
            'corr': 'Corr', 'cov': 'Cov', 'ref': 'Ref', 'delta': 'Delta',
            'slope': 'Slope', 'rank': 'Rank', 'zscore': 'ZScore'
        }
        
        for old, new in func_mapping.items():
            expr = re.sub(rf'\b{old}\b', new, expr, flags=re.IGNORECASE)
        
        return expr
    
    def _eval_expression(self, expr: str, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """递归解析和执行表达式"""
        # 处理括号表达式
        while '(' in expr and ')' in expr:
            # 找到最内层的括号
            start = expr.rfind('(')
            end = expr.find(')', start)
            
            if start == -1 or end == -1:
                break
            
            inner_expr = expr[start+1:end]
            
            # 解析函数调用或运算
            if start > 0 and (expr[start-1].isalpha() or expr[start-1].isdigit() or expr[start-1] == '_'):
                # 函数调用：向左扫描得出函数名（支持字母、数字、下划线），避免被逗号/空格/嵌套干扰
                i = start - 1
                while i >= 0 and (expr[i].isalpha() or expr[i].isdigit() or expr[i] == '_'):
                    i -= 1
                func_start = i + 1
                func_name = expr[func_start:start]
                
                # 解析参数
                params = self._parse_parameters(inner_expr, data)
                
                # 执行函数
                result = self._execute_function(func_name, params)
                
                # 替换表达式
                expr = expr[:func_start] + 'TEMP_RESULT' + expr[end+1:]
                data['TEMP_RESULT'] = result
            else:
                # 简单运算
                result = self._eval_simple_expression(inner_expr, data)
                expr = expr[:start] + 'TEMP_RESULT' + expr[end+1:]
                data['TEMP_RESULT'] = result
        
        # 处理最终表达式
        return self._eval_simple_expression(expr, data)
    
    def _parse_parameters(self, param_str: str, data: Dict) -> list:
        """解析函数参数"""
        params = []
        depth = 0
        current = ''
        
        for char in param_str + ',':
            if char == '(':
                depth += 1
                current += char
            elif char == ')':
                depth -= 1
                current += char
            elif char == ',' and depth == 0:
                params.append(current.strip())
                current = ''
            else:
                current += char
        
        # 解析每个参数
        parsed_params = []
        for param in params:
            if param in data:
                parsed_params.append(data[param])
            elif param.isdigit() or (param.replace('.', '', 1).isdigit() and param.count('.') <= 1):
                # 把整数样式的数字解析为 int，避免将 float 传给 pandas rolling window
                try:
                    if '.' in param:
                        val = float(param)
                        if val.is_integer():
                            parsed_params.append(int(val))
                        else:
                            parsed_params.append(val)
                    else:
                        parsed_params.append(int(param))
                except Exception:
                    parsed_params.append(float(param))
            else:
                # 尝试作为字段名
                field_name = self.field_aliases.get(param.lower(), param)
                if field_name in data:
                    parsed_params.append(data[field_name])
                else:
                    # 递归解析表达式
                    result = self._eval_simple_expression(param, data)
                    parsed_params.append(result)
        
        return parsed_params
    
    def _execute_function(self, func_name: str, params: list) -> pd.DataFrame:
        """执行函数"""
        if func_name not in self.operators:
            raise ValueError(f"未知算子: {func_name}")
        
        operator = self.operators[func_name]
        
        try:
            # 根据参数数量调用算子
            if len(params) == 1:
                return operator(params[0])
            elif len(params) == 2:
                # 检查第二个参数是否为数值（窗口参数），并确保传入的是整数
                if isinstance(params[1], (int, float)):
                    try:
                        window = int(params[1])
                        return operator(params[0], window)
                    except Exception:
                        return operator(params[0], params[1])
                else:
                    return operator(params[0], params[1])
            elif len(params) == 3:
                return operator(params[0], params[1], params[2])
            else:
                raise ValueError(f"不支持的参数数量: {len(params)}")
        except Exception as e:
            # 降级处理：尝试不同参数形式
            if len(params) == 2:
                try:
                    # 尝试将第二个参数转换为整数窗口值
                    if hasattr(params[1], 'iloc'):
                        window_val = params[1].iloc[0] if not params[1].empty else 10
                    else:
                        window_val = float(params[1])
                    try:
                        window_val = int(window_val)
                    except Exception:
                        pass

                    return operator(params[0], window_val)
                except:
                    return operator(params[0], params[1])
            else:
                raise e
    
    def _eval_simple_expression(self, expr: str, data: Dict) -> pd.DataFrame:
        """解析简单表达式"""
        expr = expr.strip()
        
        # 检查是否为临时结果
        if expr == 'TEMP_RESULT' and 'TEMP_RESULT' in data:
            return data['TEMP_RESULT']
        
        # 检查是否为字段名
        if expr in data:
            return data[expr]
        
        # 检查是否为字段别名
        field_name = self.field_aliases.get(expr.lower(), expr)
        if field_name in data:
            return data[field_name]
        
        # 处理二元运算
        operators = ['+', '-', '*', '/']
        for op in operators:
            if op in expr:
                parts = expr.split(op, 1)
                left = self._eval_simple_expression(parts[0].strip(), data)
                right = self._eval_simple_expression(parts[1].strip(), data)
                
                if op == '+':
                    return left + right
                elif op == '-':
                    return left - right
                elif op == '*':
                    return left * right
                elif op == '/':
                    return left / (right + 1e-10)
        
        # 如果是数值
        try:
            value = float(expr)
            # 创建一个与数据形状相同的DataFrame
            sample_data = next(iter(data.values()))
            return pd.DataFrame(value, 
                              index=sample_data.index, 
                              columns=sample_data.columns)
        except ValueError:
            raise ValueError(f"无法解析表达式: {expr}")