from typing import Callable

import torch
from torch import Tensor

from .base import Grouping
from .reduce import ReduceGroup
from .shape_utils import Shape


class CatGrouping(Grouping):
    def __init__(self, split_sizes: list[int] | None = None, dim: int = 0):
        super().__init__()
        self.split_sizes = split_sizes
        self.dim = dim

    def group(self, shape: Shape):
        return CatGroup(shape, self.split_sizes, self.dim)


class CatGroup(ReduceGroup):
    def __init__(self, shape: Shape, split_sizes: list[int], dim: int):
        super().__init__(shape.replace(dim, len(split_sizes)), (dim,))
        self.reduce_dim = None
        self.split_sizes = split_sizes
        self.dim = dim

    def _split_input_and_params(self, input: Tensor, params: list[Tensor]):
        for index, split_input in enumerate(input.split(self.split_sizes, self.dim)):
            yield split_input, [param.select(self.dim, index).unsqueeze(self.dim) for param in params]

    def apply(self, method: Callable[[Tensor, ...], Tensor], input: Tensor, params: list[Tensor], /, **kwargs):
        return torch.concat([
            super().apply(method, split_input, split_params, **kwargs)
            for split_input, split_params in self._split_input_and_params(input, params)
        ], dim=self.dim)

    def update(self, method: Callable[[Tensor, ...], None], input: Tensor, params: list[Tensor], /, **kwargs):
        for split_input, split_params in self._split_input_and_params(input, params):
            super().update(method, split_input, split_params, **kwargs)
