import torch
import torch.utils.cpp_extension
from typing import Optional
from .codegen.codegen import generate_operator_source
import uuid

class SumOperator:
    def __init__(self, nrow: int, ncol: int, expr: str):
        assert nrow == 1 and ncol == 3, f"Not Implemented for (nrow, ncol) = ({nrow}, {ncol})"

        self.nrow = nrow
        self.ncol = ncol
        
        self.wrapper_def, self.cuda_def = generate_operator_source(nrow, ncol, expr)
        
        self.compiled = False
    
    def compile(self, build_dir: Optional[str] = None, identifier: Optional[str] = None):
        if identifier is None:
            identifier = str(uuid.uuid4()).replace('-', '')
        
        if build_dir is None:
            build_dir = 'build/'

        self.module = torch.utils.cpp_extension.load_inline(name=f"custom_xmm_{identifier}_operator",
                                                            cpp_sources=[self.wrapper_def],
                                                            cuda_sources=[self.cuda_def],
                                                            build_directory=build_dir,
                                                            verbose=True)
        
        self.compiled = True

    def forward(self, *operands):
        if not self.compiled:
            raise RuntimeError("Operator Not Compiled!")

        return self.module.forward(*operands)
    
    def backward(self, *operands):
        if not self.compiled:
            raise RuntimeError("Operator Not Compiled!")

        return self.module.backward(*operands)




















