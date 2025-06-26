import functools
import inspect
from dataclasses import dataclass
from functools import wraps
from types import MethodType
from typing import Callable

import torch
from torch import Tensor
from torch.nn import Module

from iqd.utils.torch_utils import foreach_add_, foreach_mul_
from ..grouping.base import Group


def get_integral_encoding_dtype(size: int, signed: bool = False) -> torch.dtype:
    if size <= 8:
        return torch.int8 if signed else torch.uint8
    if size <= 16:
        return torch.int16 if signed else torch.uint16
    if size <= 32:
        return torch.int32 if signed else torch.uint32
    if size <= 64:
        return torch.int64 if signed else torch.uint64
    raise ValueError('storege size too large.')


@dataclass
class Calibrator:
    function: Callable


def _extract_parameters(function):
    parameters = inspect.signature(function).parameters
    for param in list(parameters.values())[1:]:
        if param.kind != inspect.Parameter.POSITIONAL_ONLY:
            break
        yield param.name


def inject(func):
    params = inspect.signature(func).parameters
    param_names = []

    for param in list(params.values())[2:]:
        match param.kind:
            case inspect.Parameter.POSITIONAL_ONLY:
                param_names.append(param.name)
            case inspect.Parameter.POSITIONAL_OR_KEYWORD:
                param_names.append(param.name)

    wrapper = with_param_unwrap(func)

    @functools.wraps(func)
    def apply(self, input, /, **kwargs):
        params = list(getattr(self, name) for name in param_names)
        return self.group.apply(wrapper.__get__(self), input, params, **kwargs)

    return apply


def with_param_unwrap(func: Callable):
    @wraps(func)
    def calibrate(self, input, params, /, **kwargs):
        return func(self, input, *params, **kwargs)

    return calibrate


calibrate = Calibrator


class QMethod:
    def __call__(self, group: Group, dtype: torch.dtype, device: torch.device) -> "QFunction":
        raise NotImplemented


class QUniformMethod(QMethod):
    def __init__(self, nbits: int, signed: bool):
        super().__init__()
        self.nbits = nbits
        self.signed = signed
        self.nlevel = (1 << nbits) - 1

        if signed:
            lower, upper = -(1 << nbits - 1), (1 << nbits - 1) - 1
        else:
            lower, upper = 0, (1 << nbits) - 1

        self.lower = lower
        self.upper = upper
        self.dtype = get_integral_encoding_dtype(nbits, signed)


class QFunction(Module):
    group: Group
    dtype: torch.dtype | None
    __calibrators: dict[str, MethodType]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        calibrators = {}

        for name, attr in cls.__dict__.items():
            if isinstance(attr, Calibrator):
                calibrators[name] = with_param_unwrap(attr.function)
        for name in calibrators.keys():
            delattr(cls, name)

        if len(calibrators) > 0:
            cls.__calibrators = { **getattr(cls, '__calibrators', {}), **calibrators }

        cls.forward = inject(cls.quantize)

    def __init__(self, group: Group, dtype: torch.dtype | None = None):
        super().__init__()
        self.group = group
        self.dtype = dtype

    def quantize(self, *args, **kwargs) -> Tensor:
        raise NotImplemented

    def encode(self, input, /) -> Tensor:
        raise NotImplemented

    def decode(self, input, /) -> Tensor:
        raise NotImplemented

    @staticmethod
    def _calibrate_ema(input: Tensor, params: tuple[Tensor, ...], /, *,
                       ema_decay: float = 0.9,
                       ema_step: int = 0.0,
                       _function,
                       **kwargs):
        if ema_step > 0:
            params_now = [torch.empty_like(param) for param in params]
            _function(input, params_now, **kwargs)
            foreach_mul_(params, ema_decay)
            foreach_add_(params, params_now, alpha=1-ema_decay)
        else:
            _function(input, params, **kwargs)

    def _load_calibrator(self, name: str):
        func = self.__calibrators.get(name, None)
        if func is not None:
            func = func.__get__(self)
        return func

    def _calibrate(self, input: Tensor, params: list[Tensor], /, mode: str, **kwargs):
        func = self._load_calibrator(mode)

        if func is None:
            raise ValueError(f"Unsupported calibration mode: {mode}")

        return func(input, params, **kwargs)

    @torch.no_grad()
    def calibrate(self, input: Tensor, /):
        kwargs = getattr(self, 'calibration', None)
        if kwargs is not None:
            kwargs = dict(kwargs)
            mode = kwargs.pop('mode')

            if mode.startswith('ema-'):
                base_mode = mode[len('ema-'):]
                func = self._load_calibrator(base_mode)
                if func is not None:
                    params = [getattr(self, name) for name in _extract_parameters(func)]
                    self.group.update(self._calibrate_ema, input, params, _function=func, **kwargs)
                    return
            else:
                func = self._load_calibrator(mode)
                params = [getattr(self, name) for name in _extract_parameters(func)]
                self.group.update(func, input, params, **kwargs)
                return

            raise ValueError(f"Unsupported calibration mode: {mode}")



class QUniformFunction(QFunction):
    dtype: torch.dtype

    def __init__(self, method: QUniformMethod, group: Group):
        super().__init__(group, method.dtype)
        self.nbits = method.nbits
        self.signed = method.signed
        self.nlevel = method.nlevel
        self.lower = method.lower
        self.upper = method.upper

    def measure_minmax(self, x: Tensor, step: Tensor, bias: Tensor,
                       x_min: Tensor | None = None, x_max: Tensor | None = None, /,
                       step_scale: float = 1.0, center: str | float = 'mean'):
        x_min = self.group.reduce(torch.amin, x, out=x_min)
        x_max = self.group.reduce(torch.amax, x, out=x_max)
        x_delta = x_max - x_min
        match center:
            case 'mean':
                x_c = self.group.reduce(torch.mean, x)
                a_c = (x_c - x_min) / x_delta
            case 'zero':
                x_c = torch.zeros_like(x_min)
                a_c = (x_c - x_min) / x_delta
            case 'mid':
                x_c = (x_max + x_min) / 2
                a_c = 0.5
            case 'min':
                x_c = x_min
                a_c = 0.0
            case _:
                x_c = x_min * center + x_max * (1 - center)
                a_c = center
        i_c = self.lower + self.nlevel * a_c
        torch.mul(x_delta, step_scale / self.nlevel, out=step).clamp_min_(self.eps)
        torch.sub(x_c, step * i_c, out=bias)
        return x_c, i_c