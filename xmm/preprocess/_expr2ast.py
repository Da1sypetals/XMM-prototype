import ast
import sympy
from sympy import Mul

from ._meta import Metadata as meta

# Define a list of supported function calls


def expr2ast(expr):
    """
    Parses the given expression to an AST and checks for supported function calls.
    """
    tree = ast.parse(expr, mode='eval')
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if node.func.id not in meta.supported_calls:
                raise ValueError(f"Unsupported function call found: '{node.func.id}'")
    return tree







