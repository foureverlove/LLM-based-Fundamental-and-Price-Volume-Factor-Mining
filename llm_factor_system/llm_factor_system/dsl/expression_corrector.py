# ==================== dsl/expression_corrector.py ====================
"""
因子表达式修正器
"""
import ast
import re
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

class ExpressionCorrector:
    """因子表达式修正器"""
    
    def __init__(self, operator_lib: Dict, field_info: Dict):
        """
        Args:
            operator_lib: 算子库 {算子名: 函数}
            field_info: 字段信息 {字段名: {'type': 'price'|'fundamental', 'frequency': 'daily'|'quarterly'}}
        """
        self.operator_lib = operator_lib
        self.field_info = field_info
        self.fundamental_fields = {
            'NET_CASH_FLOWS_OPER_ACT', 'TOT_ASSETS', 'CASH_RECP_SG_AND_RS',
            'CASH_PAY_GOODS_PURCH_SERV_REC', 'Net_Profit_Excl_Min_Int_Inc',
            'Tot_Cur_Liab', 'MarketValue'
        }
    
    def correct(self, expression: str) -> str:
        """修正因子表达式"""
        try:
            # 1. 预处理：移除多余空格、window等
            expr = self._preprocess(expression)
            
            # 2. 解析AST
            tree = ast.parse(expr)
            
            # 3. 类型推断和修正
            corrected_tree = self._visit_and_correct(tree)
            
            # 4. 生成修正后的表达式
            corrected_expr = ast.unparse(corrected_tree)
            
            # 5. 数据对齐修正
            corrected_expr = self._align_data_frequency(corrected_expr)
            
            return corrected_expr
            
        except Exception as e:
            # 如果修正失败，返回原始表达式并记录错误
            print(f"表达式修正失败: {e}")
            return expression
    
    def _preprocess(self, expr: str) -> str:
        """预处理表达式"""
        # 移除window字样
        expr = re.sub(r'window\s*=\s*\d+', '', expr)
        
        # 标准化运算符名
        operator_mapping = {
            'mean': 'Mean',
            'std': 'Std',
            'corr': 'Corr',
            'cov': 'Cov',
            'var': 'Var',
            'ema': 'EMA',
            'slope': 'Slope'
        }
        
        for old, new in operator_mapping.items():
            expr = re.sub(rf'\b{old}\b', new, expr, flags=re.IGNORECASE)
        
        return expr
    
    def _visit_and_correct(self, node: ast.AST) -> ast.AST:
        """遍历AST并修正节点"""
        if isinstance(node, ast.Call):
            # 修正函数调用
            return self._correct_call(node)
        elif isinstance(node, ast.BinOp):
            # 检查二元运算的量纲一致性
            return self._check_dimension_consistency(node)
        
        # 递归处理子节点
        for field in node._fields:
            value = getattr(node, field, None)
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, ast.AST):
                        value[i] = self._visit_and_correct(item)
            elif isinstance(value, ast.AST):
                setattr(node, field, self._visit_and_correct(value))
        
        return node
    
    def _correct_call(self, node: ast.Call) -> ast.Call:
        """修正函数调用"""
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        
        if func_name:
            # 模糊匹配运算符名
            matched_op = self._fuzzy_match_operator(func_name)
            if matched_op and matched_op != func_name:
                node.func.id = matched_op
            
            # 检查参数数量
            expected_args = self._get_expected_args(matched_op or func_name)
            if expected_args is not None and len(node.args) != expected_args:
                # 修正参数数量
                node.args = self._fix_arguments(node.args, expected_args)
        
        # 递归处理参数
        node.args = [self._visit_and_correct(arg) for arg in node.args]
        
        return node
    
    def _fuzzy_match_operator(self, op_name: str) -> str:
        """模糊匹配运算符名"""
        op_name_lower = op_name.lower()
        for op in self.operator_lib.keys():
            if op.lower() == op_name_lower:
                return op
        
        # 相似度匹配
        for op in self.operator_lib.keys():
            if self._similarity(op_name, op) > 0.8:
                return op
        
        return op_name
    
    def _similarity(self, s1: str, s2: str) -> float:
        """计算字符串相似度"""
        s1, s2 = s1.lower(), s2.lower()
        if s1 == s2:
            return 1.0
        
        # 简单编辑距离相似度
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                dp[i][j] = min(
                    dp[i-1][j] + 1,      # 删除
                    dp[i][j-1] + 1,      # 插入
                    dp[i-1][j-1] + cost  # 替换
                )
        
        max_len = max(m, n)
        return 1 - dp[m][n] / max_len if max_len > 0 else 0
    
    def _check_dimension_consistency(self, node: ast.BinOp) -> ast.AST:
        """检查二元运算的量纲一致性"""
        # 这里可以添加量纲检查逻辑
        # 例如：检查价格和变化率是否直接运算
        return node
    
    def _align_data_frequency(self, expr: str) -> str:
        """对齐数据频率"""
        # 对于基本面因子，自动添加日期对齐
        # 注意：实际的数据对齐应该在executor中处理，这里只做标记
        for field in self.fundamental_fields:
            if field in expr:
                # 基本面数据需要时间对齐，但具体对齐逻辑在executor中实现
                # 这里可以添加注释或标记
                pass
        
        return expr
    
    def _get_expected_args(self, op_name: str) -> Optional[int]:
        """获取运算符期望的参数数量"""
        # 常见算子的参数数量
        arg_counts = {
            'Mean': 2, 'Std': 2, 'Max': 2, 'Min': 2, 'EMA': 2, 'SMA': 2,
            'Ref': 2, 'Delta': 2, 'Slope': 2, 'Corr': 3, 'Cov': 3,
            'Var': 2, 'ZScore': 2, 'Rank': 1, 'Log': 1, 'Abs': 1,
            'Sign': 1, 'Power': 2, 'Sqrt': 1, 'CSZScore': 1,
            'CSNormalize': 1, 'YoYChg': 2, 'YoYVar': 2, 'MomLastZScore': 2,
            'WMA': 2, 'PctChange': 2, 'TSRank': 2, 'TSMin': 2, 'TSMax': 2
        }
        
        op_name_lower = op_name.lower()
        for op, count in arg_counts.items():
            if op.lower() == op_name_lower:
                return count
        
        # 默认返回None（不限制）
        return None
    
    def _fix_arguments(self, args: List[ast.AST], expected_count: Optional[int]) -> List[ast.AST]:
        """修正参数数量"""
        if expected_count is None:
            return args
        
        if len(args) == expected_count:
            return args
        
        # 如果参数过多，截断
        if len(args) > expected_count:
            return args[:expected_count]
        
        # 如果参数过少，补充默认值
        while len(args) < expected_count:
            args.append(ast.Constant(value=10))  # 默认窗口为10
        
        return args