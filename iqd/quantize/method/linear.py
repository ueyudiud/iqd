import numpy as np
import torch
from torch import Tensor
from torch.nn import Parameter

from .base import calibrate, QUniformMethod, QUniformFunction, inject
from ..functional import lsq2
from ..grouping.base import Group


class QLinearMethod(QUniformMethod):
    def __init__(self, nbits: int, eps: float = 1e-9):
        super().__init__(nbits, False)
        self.eps = eps

    def __call__(self, group: Group, dtype: torch.dtype, device: torch.device):
        return QLinear(self, group, dtype, device)


class QLinear(QUniformFunction):
    def __init__(self, module: QLinearMethod, group: Group, dtype: torch.dtype, device: torch.device):
        super().__init__(module, group)
        self.eps = module.eps
        self.step = Parameter(torch.empty(group.shape, dtype=dtype, device=device))
        self.bias = Parameter(torch.empty(group.shape, dtype=dtype, device=device))

    def extra_repr(self):
        return f"nbits={self.nbits}, eps={self.eps}"

    def quantize(self, x: Tensor, step: Tensor, bias: Tensor):
        return lsq2(x, step, bias, self.lower, self.upper)

    @inject
    def encode(self, x: Tensor, step: Tensor, bias: Tensor) -> Tensor:
        x = torch.round((x - bias) / step)
        x = x.clamp_(self.lower, self.upper).type(self.dtype)
        return x

    @inject
    def decode(self, x: Tensor, step: Tensor, bias: Tensor) -> Tensor:
        x = x.type(step.dtype)
        return x.mul_(step).add_(bias)

    @calibrate
    def minmax(self, x: Tensor, step: Tensor, bias: Tensor, /,
               step_scale: float = 1.0, center: str | float = 'mean'):
        return self.measure_minmax(x, step, bias, step_scale=step_scale, center=center)

    @calibrate
    def saturate(self, x: Tensor, step: Tensor, bias: Tensor, /,
                 init_mode: str = 'minmax', center: str | float = 'mean',
                 step_scale_range: tuple[float, float] = (0.2, 1.0),
                 bias_scale_range: tuple[float, float] = (-0.5, 0.5),
                 num_step_candidate: int = 80,
                 num_bias_candidate: int = 1,
                 norm_power: float = 2.4,
                 ):
        assert init_mode in ('minmax',), ValueError(f"Unsupported initial calibrate mode {init_mode}")

        init_step = torch.empty_like(step)
        init_bias = torch.empty_like(bias)

        x_c, i_c = self._calibrate(x, [init_step, init_bias], mode=init_mode, center=center)

        best_step = step
        best_bias = bias
        best_score = torch.full_like(step, float('inf'))

        step_scales = np.linspace(step_scale_range[0], step_scale_range[1], num_step_candidate)

        if num_bias_candidate > 1:
            for step_scale in step_scales:
                step = init_step * step_scale

                for bias_scale in np.linspace(bias_scale_range[0], bias_scale_range[1], num_bias_candidate):
                    bias = x_c + step * (bias_scale * self.nlevel + self.lower - i_c)

                    x_q = self.quantize(x, step, bias)
                    score = self.group.reduce(torch.sum, (x - x_q).abs_().pow_(norm_power))

                    mask = score < best_score

                    best_step[mask] = step[mask]
                    best_bias[mask] = bias[mask]
                    best_score[mask] = score[mask]
        else:
            for step_scale in step_scales:
                step = init_step * step_scale
                bias = x_c + step * (self.lower - i_c)

                x_q = self.quantize(x, step, bias)
                score = self.group.reduce(torch.sum, (x - x_q).abs_().pow_(norm_power))

                mask = score < best_score

                best_step[mask] = step[mask]
                best_bias[mask] = bias[mask]
                best_score[mask] = score[mask]
