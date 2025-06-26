from typing import Callable

from torch import Tensor

from .base import Grouping, Group


class GroupGrouping(Grouping):
    def __init__(self, base: Grouping, dim: int, num: int):
        self.base = base
        self.dim = dim
        self.num = num

    def group(self, shape):
        group = self.base.group(shape.unflat(self.dim, (-1, self.num)))
        return GroupGroup(group, self.dim, self.num)


class GroupGroup(Group):
    def __init__(self, base: Group, dim: int, num: int):
        super().__init__(base.shape)
        self.base = base
        self.dim = dim
        self.num = num

    def reduce(self, method: Callable[[Tensor, ...], Tensor], input: Tensor, /, *args, **kwargs) -> Tensor:
        return self.base.reduce(method, input.unflatten(self.dim, (-1, self.num)), *args, *kwargs)

    def apply(self, method: Callable[[Tensor, ...], Tensor], input: Tensor, params: list[Tensor], /, **kwargs):
        shape = input.shape
        return self.base.apply(method, input.unflatten(self.dim, (-1, self.num)), params, **kwargs).reshape(shape)

    def update(self, method: Callable[[Tensor, ...], None], input: Tensor, params: list[Tensor], /, **kwargs):
        self.base.update(method, input.unflatten(self.dim, (-1, self.num)), params, **kwargs)