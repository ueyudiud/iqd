import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.parameter import Parameter

from .modeling_utils import empty_tensor
from .module import ModuleQat
from ..method.base import QFunction


class LeafQ(Module):
    value: Tensor

    def __init__(self, function: QFunction, shape, dtype, device, out: Tensor | None = None):
        super().__init__()
        self.function = function
        self.register_buffer('value', empty_tensor(shape, dtype=function.dtype or dtype, device=device, out=out))

    def forward(self) -> Tensor:
        return self.function.decode(self.value)


class PseudoLeafQ(ModuleQat):
    def __init__(self, function: QFunction, origin: Tensor):
        super().__init__()
        self.function = function
        self.origin = Parameter(origin)

    @property
    def value(self):
        return self.function.encode(self.origin)

    def forward(self):
        return self.function(self.origin)


class MaskedLeafQ(LeafQ, ModuleQat):
    origin: Tensor
    mask: Tensor

    def __init__(self, function: QFunction, origin: Tensor):
        super().__init__(function, origin.shape, origin.dtype, origin.device)
        self.register_buffer('mask', torch.zeros_like(origin, dtype=torch.bool))
        self.origin = Parameter(origin)

    def forward(self) -> Tensor:
        return torch.where(self.mask, self.function.decode(self.value), self.origin)

    def detach(self):
        return LeafQ(self.function, self.value.shape, self.value.dtype, self.value.device, self.value)