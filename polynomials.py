from xmm.SumOperator import SumOperator
import torch
import torch.nn as nn

expression = "r1 * c1 + r1 * r1 * c2 + r1 * r1 * r1 * c3 + c4"
op = SumOperator(1, 4, expression)
op.compile(identifier="operator_poly")

class XmmFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, c1, c2, c3, bias):

        ctx.save_for_backward(x, c1, c2, c3, bias)

        return op.forward(x, c1, c2, c3, bias)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, c1, c2, c3, bias = ctx.saved_tensors

        grad_x, grad_c1, grad_c2, grad_c3, grad_bias = op.backward(grad_output, x, c1, c2, c3, bias)

        return grad_x, grad_c1, grad_c2, grad_c3, grad_bias

class XmmLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.bias = nn.Parameter(torch.zeros(out_features, in_features))

        self.c1 = nn.Parameter(torch.empty(out_features, in_features))
        self.c2 = nn.Parameter(torch.empty(out_features, in_features))
        self.c3 = nn.Parameter(torch.empty(out_features, in_features))

        nn.init.kaiming_uniform_(self.c1, a=(5 ** .5))
        nn.init.kaiming_uniform_(self.c2, a=(5 ** .5))
        nn.init.kaiming_uniform_(self.c3, a=(5 ** .5))

        # Batch normalization
        self.bn = nn.BatchNorm1d(out_features)


    def forward(self, x):

        x = XmmFn.apply(x, self.c1, self.c2, self.c3, self.bias)

        return self.bn(x)
    

