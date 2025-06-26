from torch import Tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from .mixin import UnaryLinearQ
from .modeling_utils import empty_tensor
from ..setting import WMap, AMap


class LinearQ(UnaryLinearQ):
    __constants__ = [
        'in_features',
        'out_features',
    ]

    def __init__(self, in_features, out_features, bias=True, weight=None, device=None, dtype=None, *,
                 wmap: WMap, amap: AMap):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.w_layer = wmap(empty_tensor((out_features, in_features), dtype=dtype, device=device, out=weight))
        if isinstance(bias, Tensor) or bias:
            self.bias = Parameter(empty_tensor(out_features, dtype=dtype, device=device, out=bias))
        else:
            self.register_parameter('bias', None)
        self.a_layer = amap((..., in_features), dtype=dtype, device=device)

    def _forward(self, input: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        return F.linear(input, weight, bias)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
