import torch
import torch.utils.cpp_extension
from typing import Optional
from .codegen.codegen import generate_operator_source
import uuid
import re
import os

def valid_identifier(identifier):
    pattern = '^[A-Za-z0-9_]+$'
    return re.match(pattern, identifier)


class SumOperator:
    def __init__(self, nrow: int, ncol: int, expr: str):
        assert nrow == 1 and ncol == 3, f"Not Implemented for (nrow, ncol) = ({nrow}, {ncol})"

        self.nrow = nrow
        self.ncol = ncol
        
        self.wrapper_def, self.cuda_def = generate_operator_source(nrow, ncol, expr)
        self.expression = expr
        
        self.compiled = False
    
    def compile(self, build_dir: Optional[str] = None, identifier: Optional[str] = None):
        if identifier is None:
            identifier = str(uuid.uuid4()).replace('-', '')
        assert valid_identifier(identifier), f"Invalid Identifier: {identifier}, pattern required: ^[A-Za-z0-9_]+$"
        
        if build_dir is None:
            build_dir = f'build_{identifier}/'
        
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)
            print('Build path does not exist, creating...')

        with open(os.path.join(build_dir, 'expression.txt'), "w") as file:
            file.write(self.expression)  

        print(f'\n[xmm] ------------------------------ Compiling Operator: {identifier} ------------------------------\n')

        self.module = torch.utils.cpp_extension.load_inline(name=f"xmm_operator_{identifier}",
                                                            cpp_sources=[self.wrapper_def],
                                                            cuda_sources=[self.cuda_def],
                                                            build_directory=build_dir,
                                                            verbose=True)
        
        print(f'\n[xmm] ------------------------------ Compilation Done: {identifier} ------------------------------\n')

        
        self.compiled = True

    def forward(self, *operands):
        if not self.compiled:
            raise RuntimeError("Operator Not Compiled!")

        return self.module.forward(*operands)
    
    def backward(self, *operands):
        if not self.compiled:
            raise RuntimeError("Operator Not Compiled!")

        return self.module.backward(*operands)




















