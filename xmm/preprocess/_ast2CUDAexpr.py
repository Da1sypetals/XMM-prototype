import ast
from ._meta import Metadata as meta

def ast2CUDAexpr(node):
    """
    Recursively converts an AST node into a Tiachi expression string.
    """
    if isinstance(node, ast.Num):  # Handle Num for Python versions < 3.8
        str_val = str(node.n)
        return str_val + 'f' if '.' in str_val else str_val + '.0f' 
    elif isinstance(node, ast.Constant):  # Handle Constant for Python versions >= 3.8
        str_val = str(node.value)
        return str_val + 'f' if '.' in str_val else str_val + '.0f' 
    elif isinstance(node, ast.Name): 
        return node.id
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return f'(-{ast2CUDAexpr(node.operand)})'
    elif isinstance(node, ast.BinOp):
        left = ast2CUDAexpr(node.left)
        right = ast2CUDAexpr(node.right)
        op_map = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/', ast.Pow: '**'}
        # print(node)
        op_symbol = op_map[type(node.op)]
        return f'({left} {op_symbol} {right})'
    elif isinstance(node, ast.Call):
        if node.func.id not in meta.supported_calls:
            raise ValueError(f'Function {node.func.id} is not supported.')
        taichi_primitive = meta.CUDA_fn_map[node.func.id]
        args = [ast2CUDAexpr(arg) for arg in node.args]
        args_str = ', '.join(args)
        return f'{taichi_primitive}({args_str})'
    else:
        raise TypeError(f'Unsupported AST node type: {type(node)}')



    