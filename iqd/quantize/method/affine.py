import numpy as np
import torch
from torch import Tensor
from torch.nn.parameter import Parameter

from .base import calibrate, QUniformMethod, QUniformFunction, inject
from ..functional import lsq2i
from ..grouping.base import Group


class QAffineMethod(QUniformMethod):
    def __init__(self, nbits: int, signed: bool = True, eps: float = 1e-9):
        super().__init__(nbits, signed)
        self.eps = eps

    def __call__(self, group: Group, dtype: torch.dtype, device: torch.device):
        return QAffine(self, group, dtype, device)


class QAffine(QUniformFunction):
    def __init__(self, method: QAffineMethod, group: Group, dtype: torch.dtype, device: torch.device):
        super().__init__(method, group)
        self.eps = method.eps
        self.step = Parameter(torch.empty(group.shape, dtype=dtype, device=device))
        self.register_buffer('zero_point', torch.empty(group.shape, dtype=self.dtype, device=device))
        self.register_buffer('bias', torch.empty(group.shape, dtype=dtype, device=device), persistent=False)

    def extra_repr(self):
        return f"nbits={self.nbits}, eps={self.eps}"

    def quantize(self, x: Tensor, step: Tensor, zero_point: Tensor) -> Tensor:
        return lsq2i(x, step, zero_point, self.lower, self.upper)

    @inject
    def encode(self, x: Tensor, step: Tensor, zero_point: Tensor) -> Tensor:
        x = x / step - zero_point
        x = x.clamp_(self.lower, self.upper).round_()
        x = x.type(self.dtype)
        return x

    @inject
    def decode(self, x: Tensor, step: Tensor, zero_point: Tensor) -> Tensor:
        x = (x + zero_point).type(step.dtype)
        return x.mul_(step)

    @calibrate
    def minmax(self, x: Tensor, step: Tensor, zero_point: Tensor, bias: Tensor, /,
               step_scale: float = 1.0, center: str | float = 'mean'):
        x_c, i_c = self.measure_minmax(x, step, bias, step_scale=step_scale, center=center)
        zero_point.copy_((bias / step).round_())
        return x_c, i_c

    @calibrate
    def saturate(self, x: Tensor, step: Tensor, zero_point: Tensor, bias: Tensor, /,
                 init_mode: str = 'minmax', center: str | float = 'mean',
                 step_scale_range: tuple[float, float] = (0.2, 1.0),
                 num_step_candidate: int = 80,
                 multi_bias_candidate: bool = False,
                 norm_power: float = 2.4,
                 ):
        assert init_mode in ('minmax',), ValueError(f"Unsupported initial calibrate mode {init_mode}")

        init_step = torch.empty_like(step)
        init_bias = torch.empty_like(bias)

        x_c, i_c = self._calibrate(x, [init_step, zero_point, init_bias], mode=init_mode, center=center)

        best_step = step
        best_zero_point = zero_point
        best_bias = bias
        best_score = torch.full_like(step, float('inf'))

        for step_scale in np.linspace(step_scale_range[0], step_scale_range[1], num_step_candidate):
            step = init_step * step_scale

            if multi_bias_candidate:
                zero_point_base_min, zero_point_base_max = torch.aminmax(x / step)
                zero_points = torch.arange(
                    max(zero_point_base_min.floor_(), self.lower),
                    min(zero_point_base_max.ceil_(), self.upper) + 1,
                    dtype=self.dtype,
                    device=best_zero_point.device,
                )
            else:
                zero_points = [(init_bias / step).round_().clamp_(self.lower, self.upper).type(self.dtype)]

            for zero_point in zero_points:
                bias = step * zero_point
                x_q = self.quantize(x, step, zero_point)

                score = self.group.reduce(torch.sum, (x - x_q).abs_().pow_(norm_power))
                mask = score < best_score

                best_step[mask] = step[mask]
                best_zero_point[mask] = zero_point[mask]
                best_bias[mask] = bias[mask]
                best_score[mask] = score[mask]

    def _calibrate_ema(self, function, x: Tensor, step: Tensor, zero_point: Tensor, bias: Tensor, /,
                       ema_decay: float = 0.9,
                       ema_step: int = 0,
                       **kwargs):
        if ema_step > 0:
            step_now = torch.empty_like(step)
            bias_now = torch.empty_like(bias)
            function(x, step_now, zero_point, bias_now, **kwargs)
            step.mul_(ema_decay).add_(step_now, alpha=1-ema_decay)
            bias.mul_(ema_decay).add_(bias_now, alpha=1-ema_decay)
            zero_point.copy_((bias / step).round().clamp_(self.upper, self.lower))
        else:
            function(x, step, zero_point, bias, **kwargs)
