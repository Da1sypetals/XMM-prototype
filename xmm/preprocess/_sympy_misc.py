import sympy

def expand_powers(expr):
    if isinstance(expr, sympy.Pow):
        base, exp = expr.args
        if exp.is_integer and exp > 0:
            return sympy.Mul(*([base] * exp), evaluate=False)
    elif expr.is_Mul:
        return expr.func(*(expand_powers(arg) for arg in expr.args), evaluate=False)
    return expr

