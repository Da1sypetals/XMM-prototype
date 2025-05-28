import ast
from ._meta import Metadata as meta


def make_power(base, exp: int):
    if exp > 2:
        return ast.BinOp(base, ast.Mult(), make_power(base, exp - 1))
    elif exp < -1:
        return ast.BinOp(make_power(base, exp + 1), ast.Div(), base)
    elif exp == -1:
        return ast.BinOp(ast.Constant(1), ast.Div(), base)
    elif exp == 0:
        return ast.Constant(1)
    elif exp == 1:
        return base
    else:  # exp == 2
        return ast.BinOp(base, ast.Mult(), base)


def sympy2ast(sympy_expr):
    """
    Converts a SymPy expression back into an AST node.
    """
    if sympy_expr.is_Add:
        # Handle addition
        args = sympy_expr.args
        left = sympy2ast(args[0])
        for term in args[1:]:
            right = sympy2ast(term)
            left = ast.BinOp(left=left, op=ast.Add(), right=right)
        return left
    elif sympy_expr.is_Mul:
        # Handle multiplication
        args = sympy_expr.args
        left = sympy2ast(args[0])
        for term in args[1:]:
            right = sympy2ast(term)
            left = ast.BinOp(left=left, op=ast.Mult(), right=right)
        return left
    elif sympy_expr.is_Pow:
        # Handle power
        base = sympy2ast(sympy_expr.base)
        exp = sympy2ast(sympy_expr.exp)
        if isinstance(exp, ast.Name):
            return ast.BinOp(left=base, op=ast.Pow(), right=exp)
        else:
            if abs(exp.value - round(exp.value)) > 1e-7:
                raise ValueError("Unsupported")
            return make_power(base, round(exp.value))
        # return ast.BinOp(left=base, op=ast.Pow(), right=exp)
    elif sympy_expr.is_Function:
        # Handle supported function calls
        func_name = sympy_expr.func.__name__
        if func_name not in meta.supported_calls:
            raise ValueError(f"Function {func_name} is not supported.")
        args = [sympy2ast(arg) for arg in sympy_expr.args]
        return ast.Call(func=ast.Name(id=func_name), args=args, keywords=[])
    elif sympy_expr.is_Symbol:
        # Handle symbols
        return ast.Name(id=str(sympy_expr))
    elif sympy_expr.is_Number:
        # Handle numbers
        return ast.Constant(value=sympy_expr.evalf())
    else:
        raise NotImplementedError(
            f"Conversion for {type(sympy_expr)} is not implemented."
        )
