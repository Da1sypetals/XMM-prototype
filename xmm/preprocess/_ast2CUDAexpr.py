import ast
from ._meta import Metadata as meta

OP_MAP = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
SIMPLE_OP = tuple(OP_MAP.keys())


def ast2CUDAexpr(node):
    """
    Recursively converts an AST node into a CUDA expression string.
    """
    if isinstance(node, ast.Num):  # Handle Num for Python versions < 3.8
        str_val = str(node.n)
        return str_val + "f" if "." in str_val else str_val + ".0f"
    elif isinstance(node, ast.Constant):  # Handle Constant for Python versions >= 3.8
        str_val = str(node.value)
        return str_val + "f" if "." in str_val else str_val + ".0f"
    elif isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return f"(-{ast2CUDAexpr(node.operand)})"

    elif isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow):
            left = ast2CUDAexpr(node.left)
            right = ast2CUDAexpr(node.right)
            # implement floating point power in CUDA-C++
            return f"powf({left}, {right})"
        elif isinstance(node.op, SIMPLE_OP):
            left = ast2CUDAexpr(node.left)
            right = ast2CUDAexpr(node.right)
            # print(node)
            op_symbol = OP_MAP[type(node.op)]
            return f"({left} {op_symbol} {right})"
        else:
            breakpoint()
            raise TypeError(f"Unsupported Binary Operation type: {type(node.op)}")
    elif isinstance(node, ast.Call):
        if node.func.id not in meta.supported_calls:
            raise ValueError(f"Function {node.func.id} is not supported.")
        taichi_primitive = meta.CUDA_fn_map[node.func.id]
        args = [ast2CUDAexpr(arg) for arg in node.args]
        args_str = ", ".join(args)
        return f"{taichi_primitive}({args_str})"
    else:
        raise TypeError(f"Unsupported AST node type: {type(node)}")
