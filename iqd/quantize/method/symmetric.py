import numpy as np
import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from .base import calibrate, QUniformMethod, QUniformFunction, inject
from ..functional import lsq
from ..grouping.base import Group


class QSymmetricMethod(QUniformMethod):
    def __init__(self, nbits: int, signed: bool = True, eps: float = 1e-9):
        super().__init__(nbits, signed)
        self.eps = eps

    def __call__(self, group: Group, dtype: torch.dtype, device: torch.device):
        return QSymmetric(self, group, dtype, device)


class QSymmetric(QUniformFunction):
    def __init__(self, module: QSymmetricMethod, group: Group, dtype: torch.dtype, device: torch.device):
        super().__init__(module, group)
        self.eps = module.eps
        self.step = Parameter(torch.empty(group.shape, dtype=dtype, device=device))

    def extra_repr(self):
        s = f"nbits={self.nbits}"
        if not self.signed:
            s += f", signed={self.signed}"
        s += f", eps={self.eps}"
        return s

    def quantize(self, x: Tensor, step: Tensor) -> Tensor:
        return lsq(x, step, self.lower, self.upper)

    @inject
    def encode(self, x: Tensor, step: Tensor) -> Tensor:
        x = torch.round(x / step)
        x = x.clamp_(self.lower, self.upper).type(self.dtype)
        return x

    @inject
    def decode(self, x: Tensor, step: Tensor) -> Tensor:
        x = x.type(step.dtype)
        return x.mul_(step)

    @calibrate
    def minmax(self, x: Tensor, step: Tensor, /,
               step_scale: float = 1.0):
        if self.signed:
            bound = self.group.reduce(torch.amax, x.abs())
            step_scale *= 2
        else:
            bound_max = self.group.reduce(torch.amax, x)
            bound_min = self.group.reduce(torch.amin, x)
            bound = torch.where(-bound_min.clamp_min(0) < bound_max.clamp_max(0), bound_min, bound_max)

        torch.mul(bound, step_scale / self.nlevel, out=step)
        step.masked_fill_(step.abs() <= self.eps, self.eps)

    @calibrate
    def saturate(self, x: Tensor, step: Tensor, /,
                 init_mode: str = 'minmax', center: str | float = 'mean',
                 step_scale_range: tuple[float, float] = (0.2, 1.0),
                 num_step_candidate: int = 80,
                 norm_power: float = 2.4,
                 ):
        assert init_mode in ('minmax',), ValueError(f"Unsupported initial calibrate mode {init_mode}")

        init_step = torch.empty_like(step)

        self._calibrate(x, [init_step], mode=init_mode, center=center)

        best_step = step
        best_score = torch.full_like(step, float('inf'))

        for step_scale in np.linspace(step_scale_range[0], step_scale_range[1], num_step_candidate):
            step = init_step * step_scale

            x_q = self.quantize(x, step)
            score = self.group.reduce(torch.sum, (x - x_q).abs_().pow_(norm_power))
            mask = score < best_score

            best_step[mask] = step[mask]
            best_score[mask] = score[mask]
