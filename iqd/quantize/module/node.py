import torch
from torch import Tensor

from .module import ModuleQat
from ..method.base import QFunction


class MaskedNodeQ(ModuleQat):
    def __init__(self, module: QFunction):
        super().__init__()
        self.module = module
        self.rate = None

    def forward(self, x: Tensor):
        y = self.module(x)
        if self.rate is not None and self.training:
            y = y.where(torch.rand_like(x) < self.rate, x)
        return y

    def detach(self):
        return self.module

    def calibrate(self, input: Tensor):
        self.module.calibrate(input)