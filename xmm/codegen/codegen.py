import ast
import sympy
from sympy import symbols, diff
from ..preprocess import expr2ast, ast2CUDAexpr, sympy2ast, ast2sympy
from sortedcontainers import SortedSet


def expected_variables(nrow, ncol):
    ss = SortedSet()
    for r in range(1, nrow + 1):
        ss.add(f'r{r}')
    for c in range(1, ncol + 1):
        ss.add(f'c{c}')
    return ss



def generate_expr(nrow, ncol, expression):
    # Step 1 & 2: Parse the expression into AST and then convert to sympy expression
    ast_tree = expr2ast(expression)
    sympy_expr = ast2sympy(ast_tree)

    # print("SymPy Expression:\n", sympy_expr)

    # Identify all identifiers (symbols) in the expression for differentiation
    identifiers = [str(s) for s in sympy_expr.free_symbols]
    identifiers_set = SortedSet(identifiers)

    ss = expected_variables(nrow, ncol)
    assert ss == identifiers_set, f"Incorrect variable use!\n\n[nrow = {nrow}, ncol = {ncol}]"\
                                    +f"\n[Expected]: {','.join(ss)}\n[Used]: {','.join(identifiers_set)}\n"\
                                    + f"[Missing]: {','.join(ss - identifiers_set)}\n[Unexpected]:{','.join(identifiers - ss)}\n"

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


# from ..templates.template_1_3.cpp import cpp_template_1_3
# from ..templates.template_1_3.cuda import cuda_template_1_3

from ..templates.cpp import generate_cpp
from ..templates.cuda import generate_cuda


def generate_operator_source(nrow: int, ncol: int, expression):

    CUDA_expr_forward, CUDA_derivatives = generate_expr(nrow, ncol, expression)

    CUDA_expr_backward = {
        str(k): v for k, v in CUDA_derivatives.items()
    }

    wrapper_def = generate_cpp(nrow, ncol)
    cuda_def = generate_cuda(nrow, ncol, CUDA_expr_forward, CUDA_expr_backward)
    
    return wrapper_def, cuda_def
    

from ..templates.optimize_r1c1.cpp import cpp_code as cpp_code_v1
from ..templates.optimize_r1c1.cuda import generate_cuda as generate_cuda_v1

def generate_operator_source_fwd_v1(nrow: int, ncol: int, expression):

    CUDA_expr_forward, CUDA_derivatives = generate_expr(nrow, ncol, expression)

    CUDA_expr_backward = {
        str(k): v for k, v in CUDA_derivatives.items()
    }

    wrapper_def = cpp_code_v1
    cuda_def = generate_cuda_v1(CUDA_expr_forward)
    
    return wrapper_def, cuda_def
    






