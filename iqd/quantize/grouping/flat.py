from typing import Callable

from torch import Tensor

from .base import Grouping, Group


class FlatGrouping(Grouping):
    def __init__(self, base: Grouping, start: int, end: int):
        self.base = base
        self.start = start
        self.end = end

    def group(self, shape):
        group = self.base.group(shape.flat(self.start, self.end))
        return FlatGroup(group, self.start, self.end)


class FlatGroup(Group):
    def __init__(self, base: Group, start: int, end: int):
        super().__init__(base.shape)
        self.base = base
        self.start = start
        self.end = end

    def reduce(self, method: Callable[[Tensor, ...], Tensor], input: Tensor, /, *args, **kwargs) -> Tensor:
        return self.base.reduce(method, input.flatten(self.start, self.end), *args, **kwargs)

    def apply(self, method: Callable[[Tensor, ...], Tensor], input: Tensor, params: list[Tensor], /, **kwargs):
        return self.base.apply(method, input.flatten(self.start, self.end), params, **kwargs)

    def update(self, method: Callable[[Tensor, ...], None], input: Tensor, params: list[Tensor], /, **kwargs):
        self.base.update(method, input.flatten(self.start, self.end), params, **kwargs)
