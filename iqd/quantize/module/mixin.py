import inspect
from dataclasses import dataclass

from torch import Tensor
from torch.nn import Module

from ..setting import WMap, AMap


@dataclass
class UnaryLinearSetting:
    wmap: WMap
    amap: AMap


class UnaryLinearQ(Module):
    w_layer: Module
    a_layer: Module

    bias: Tensor | None

    @classmethod
    def from_origin(cls, origin: Module, setting: UnaryLinearSetting, **extra):
        excludes = { 'self', 'wmap', 'amap' }
        kwargs = { 'wmap': setting.wmap, 'amap': setting.amap }
        for name in inspect.signature(cls.__init__).parameters.keys():
            if name in excludes:
                continue
            elif name == 'device':
                param = origin.weight.device
            elif name == 'dtype':
                param = origin.weight.dtype
            else:
                param = getattr(origin, name)
            kwargs[name] = param
        return cls(**kwargs)

    @property
    def weight(self) -> Tensor:
        return self.w_layer()

    def _forward(self, input: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        raise NotImplemented

    def forward(self, input: Tensor) -> Tensor:
        return self._forward(self.a_layer(input), self.weight, self.bias)
