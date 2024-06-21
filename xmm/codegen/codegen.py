import ast
import sympy
from sympy import symbols, diff
from ..preprocess import expr2ast, ast2CUDAexpr, sympy2ast, ast2sympy


def generate_expr(expression):
    # Step 1 & 2: Parse the expression into AST and then convert to sympy expression
    ast_tree = expr2ast(expression)
    sympy_expr = ast2sympy(ast_tree)

    # print("SymPy Expression:\n", sympy_expr)

    # Identify all identifiers (symbols) in the expression for differentiation
    identifiers = [str(s) for s in sympy_expr.free_symbols]

    # Step 4: Take derivatives w.r.t all identifiers and save both the sympy diff and convert back to AST
    derivative_sympy_exprs = {id: diff(sympy_expr, symbols(id)) for id in identifiers}

    # print(sympy_expr)
    # for k, v in derivative_sympy_exprs.items():
    #     print(f'{k} : {v}')

    derivative_ast_trees = {id: sympy2ast(expr) for id, expr in derivative_sympy_exprs.items()}

    # Step 5: Convert all AST expressions (original and derivatives) into Taichi expressions
    CUDA_expr = ast2CUDAexpr(ast_tree.body)
    CUDA_derivatives = {id: ast2CUDAexpr(tree) for id, tree in derivative_ast_trees.items()}

    return CUDA_expr, CUDA_derivatives


from ..templates.template_1_3.cpp import cpp_template_1_3
from ..templates.template_1_3.cuda import cuda_template_1_3


def generate_operator_source(nrow: int, ncol: int, expression):
    assert nrow == 1 and ncol == 3, f"Not Implemented for (nrow, ncol) = ({nrow}, {ncol})"
    
    CUDA_expr_forward, CUDA_derivatives = generate_expr(expression)

    CUDA_expr_backward = {
        str(k): v for k, v in CUDA_derivatives.items()
    }

    wrapper_def = cpp_template_1_3
    cuda_def = cuda_template_1_3(CUDA_expr_forward, 
                                 CUDA_expr_backward['r1'],
                                 CUDA_expr_backward['c1'],
                                 CUDA_expr_backward['c2'],
                                 CUDA_expr_backward['c3'],
                                 )
    
    return wrapper_def, cuda_def






