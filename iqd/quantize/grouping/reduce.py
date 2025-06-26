from typing import Callable, Iterable

from torch import Tensor

from .base import Grouping, Group


class ReduceGrouping(Grouping):
    dim: tuple[int, ...]

    def __init__(self, *dim: int):
        self.dim = dim

    def group(self, shape: tuple[int, ...]):
        return ReduceGroup(shape, self.dim)


class ReduceGroup(Group):
    reduce_dim: tuple[int, ...]

    def __init__(self, shape: tuple[int, ...], dim: Iterable[int] | None = None):
        shape2 = [1] * len(shape)
        if dim is not None:
            for d in dim:
                assert shape[d] > 0, ValueError("The channel-wise keep dimension cannot have variable size.")
                shape2[d] = shape[d]
        super().__init__(shape2)
        self.reduce_dim = tuple(i for i, s in enumerate(shape2) if s == 1)

    def reduce(self, method: Callable[[Tensor, ...], Tensor], input: Tensor, /, *args, **kwargs):
        return method(input, *args, dim=self.reduce_dim, keepdim=True, **kwargs)