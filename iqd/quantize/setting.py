from typing import Callable

import torch
from torch import Tensor
from torch.nn import Module

from .grouping import Grouping
from .method import QMethod

type AMap = Callable[..., Module]

type WMap = Callable[[Tensor], Module]


def amap(method: QMethod, grouping: Grouping, calibration = None) -> AMap:
    def apply(shape: tuple[int, ...], dtype: torch.dtype | None, device: torch.device | None):
        function = method(grouping(*shape), dtype, device)
        function.calibration = calibration
        return function

    return apply


def wmap(method: QMethod, grouping: Grouping, calibration = None) -> WMap:
    from .module.leaf import LeafQ

    def apply(value: Tensor, /):
        function = method(grouping(*value.shape), value.dtype, value.device)
        function.calibration = calibration
        return LeafQ(function, value.shape, value.dtype, value.device)

    return apply

