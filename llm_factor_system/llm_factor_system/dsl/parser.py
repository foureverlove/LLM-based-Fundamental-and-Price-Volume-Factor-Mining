import ast

class DSLParser(ast.NodeVisitor):

    def __init__(self, operators, variables):
        self.operators = operators
        self.variables = variables

    def visit_Call(self, node):
        if not isinstance(node.func, ast.Name):
            raise ValueError("非法函数调用")

        if node.func.id not in self.operators:
            raise ValueError(f"不支持算子: {node.func.id}")

        for arg in node.args:
            self.visit(arg)

    def visit_Name(self, node):
        if node.id not in self.variables and not node.id.isnumeric():
            raise ValueError(f"未知变量: {node.id}")

    def check(self, expr: str):
        tree = ast.parse(expr)
        self.visit(tree)
