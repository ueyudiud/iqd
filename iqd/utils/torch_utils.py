import torch
from torch import Tensor
from typing_extensions import Iterable

if hasattr(torch, '_foreach_add_'):
    foreach_add_ = torch._foreach_add_
else:
    def foreach_add_(xs: Iterable[Tensor], ys: list[Tensor] | float, /, *, alpha = 1):
        if isinstance(ys, Iterable):
            for x, y in zip(xs, ys):
                x.add_(y, alpha=alpha)
        else:
            for x in xs:
                x.add_(ys, alpha=alpha)
        return xs

if hasattr(torch, '_foreach_mul'):
    foreach_mul_ = torch._foreach_mul_
else:
    def foreach_mul_(xs: Iterable[Tensor], ys: list[Tensor] | float, /):
        if isinstance(ys, Iterable):
            for x, y in zip(xs, ys):
                x.mul_(y)
        else:
            for x in xs:
                x.mul_(ys)
        return xs
