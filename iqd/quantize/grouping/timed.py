from typing import Callable

import torch
from torch import Tensor

from iqd.utils import globals
from .base import Grouping, Group
from .shape_utils import Shape


class TimedGrouping(Grouping):
    base: Grouping
    num_step: int | None

    def __init__(self, base: Grouping | None, num_step: int | None = None):
        self.base = base
        self.num_step = num_step

    def group(self, shape: Shape):
        group = self.base.group(shape)
        return TimedGroup(group, self.num_step or globals.total_timestep)


class TimedGroup(Group):
    def __init__(self, base: Group, num_step: int):
        assert base.shape[0] < 0 or base.shape[0] == 1
        super().__init__((num_step, *base.shape[1:]))
        self.base = base
        self.num_step = num_step

    def reduce(self, method: Callable, input: Tensor, /, *args, **kwargs) -> Tensor:
        return self.base.reduce(method, input, *args, **kwargs)

    def apply(self, method: Callable, input: Tensor, params: list[Tensor], /, **kwargs):
        scaled_timestep = globals.current_timestep * self.num_step
        index = scaled_timestep // globals.total_timestep
        index_next = (index + 1).clamp_max(self.num_step - 1)
        delta = (scaled_timestep - index * globals.total_timestep) / globals.total_timestep
        params = list(p[index] * (1. - delta) + p[index_next] * delta for p in params)
        return self.base.apply(method, input, params, **kwargs)

    def update(self, method: Callable, input: Tensor, params: list[Tensor], /, **kwargs):
        assert globals.current_timestep.numel() == 1
        params = list(p[globals.stored_timestep].unsqueeze(0) for p in params)
        self.base.update(method, input, params, **kwargs)
