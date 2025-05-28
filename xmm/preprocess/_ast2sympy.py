import ast
from ._meta import Metadata as meta
import sympy


def create_symbols(node):
    """
    Recursively traverses an AST node to identify symbols and create them using sympy.
    """
    if isinstance(node, ast.Name):
        return sympy.symbols(node.id)
    elif isinstance(node, ast.Constant):
        return node.n
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -create_symbols(node.operand)
    elif isinstance(node, ast.BinOp):
        left = create_symbols(node.left)
        right = create_symbols(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            # Use sympy.Mul with evaluate=False for multiplication to prevent automatic simplification
            return sympy.Mul(left, right, evaluate=False)
        elif isinstance(node.op, ast.Div):
            return left / right
        elif isinstance(node.op, ast.Pow):
            return left**right
    elif isinstance(node, ast.Call):
        if node.func.id not in meta.supported_calls:
            raise ValueError(f"Function {node.func.id} is not supported.")
        args = [create_symbols(arg) for arg in node.args]
        return meta.call_map[node.func.id](*args)


def ast2sympy(ast_tree):
    """
    Generates a sympy expression from the given AST tree.
    """
    return create_symbols(ast_tree.body)
