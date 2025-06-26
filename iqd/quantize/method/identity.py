import torch
from torch import Tensor

from .base import QMethod, QFunction, calibrate
from ..grouping.base import Group
from ..grouping.full import FullGroup


class QIdentityMethod(QMethod):
    def __call__(self, group: Group, dtype: torch.dtype, device: torch.device):
        return QIdentity()


class QIdentity(QFunction):
    def __init__(self):
        super().__init__(FullGroup())

    def quantize_(self, input: Tensor, /) -> Tensor:
        return input

    def encode_(self, input: Tensor, /) -> Tensor:
        return input

    def decode_(self, input: Tensor, /) -> Tensor:
        return input

    @calibrate
    def none(self, input: Tensor):
        pass
