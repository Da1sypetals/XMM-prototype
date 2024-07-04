from xmm.SumOperator_v1 import SumOperator_v1
import torch
import torch.nn as nn

expression = "r1 * c1"
op = SumOperator_v1(1, 1, expression)
op.compile(identifier="operator_v0")

class XmmFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):

        ctx.save_for_backward(x, weight)

        return op.forward(x, weight)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors

        grad_x, grad_weight = op.backward(grad_output, x, weight)

        return grad_x, grad_weight

class XmmLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=(5 ** .5))

        self.bn = nn.BatchNorm1d(out_features)


    def forward(self, x):

        x = XmmFn.apply(x.contiguous(), self.weight)

        return self.bn(x)
    

