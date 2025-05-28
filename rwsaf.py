from xmm.SumOperator import SumOperator
import torch
import torch.nn as nn

# expression = "0.867325070 * c3 * (c1 * c1 * (r1 + c2) * (r1 + c2) - 1) * exp(-0.5 * c1 * c1 * (r1 + c2) * (r1 + c2))"
expression = "c3 * (c1 * (r1 + c2) - 1) * exp(-0.5 * c1 * c1 * (r1 + c2) * (r1 + c2))"
# expression = "c3 * exp(-c1 * r1 * r1) + c2"
op = SumOperator(1, 3, expression)
op.compile(identifier="operator_v0")


class XmmFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, bias, weight):
        ctx.save_for_backward(x, scale, bias, weight)

        return op.forward(x, scale, bias, weight)

    @staticmethod
    def backward(ctx, grad_output):
        x, scale, bias, weight = ctx.saved_tensors

        grad_x, grad_scale, grad_bias, grad_weight = op.backward(
            grad_output, x, scale, bias, weight
        )

        return grad_x, grad_scale, grad_bias, grad_weight


class XmmLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features, in_features))

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=(5**0.5))

        self.base_activation = nn.SiLU()

        # Batch normalization
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = XmmFn.apply(x, self.scale, self.bias, self.weight)

        return self.bn(x)
