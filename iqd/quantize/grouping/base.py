from typing import Callable, Iterable

import torch
from torch import Tensor

from .shape_utils import Shape


class Grouping:
    def group(self, shape: Shape):
        raise NotImplemented

    def __call__(self, *shape) -> "Group":
        return self.group(Shape(shape))


class Group:
    shape: torch.Size

    def __init__(self, shape: Iterable[int]):
        self.shape = torch.Size(shape)

    def reduce(self, method: Callable, input: Tensor, /, *args, **kwargs) -> Tensor:
        return method(input, *args, **kwargs)

    def apply(self, method: Callable, input: Tensor, params: list[Tensor], /, **kwargs):
        return method(input, params, **kwargs)

    def update(self, method: Callable, input: Tensor, params: list[Tensor], /, **kwargs):
        if len(params) > 0 and len(input.shape) != len(params[0].shape):
            input = input.flatten(0, len(input.shape) - len(params[0].shape))
        method(input, params, **kwargs)
