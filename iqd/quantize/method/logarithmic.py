# import numpy as np
# import torch
# from torch import Tensor
# from torch.autograd import Function
# from torch.autograd.function import FunctionCtx
# from torch.nn import Parameter
#
# from iqd.utils.dtypes import ceil_storage_dtype
# from .base import QMethod, _BaseContext
#
# _SQRT_2_2 = 0.70710678118654752440084436210485
# _LOG2 = 0.69314718055994530941723212145818
#
#
# class LogarithmicQuant(Function):
#     @staticmethod
#     def forward(ctx: FunctionCtx, input: Tensor, bias: Tensor, min_exp: int) -> Tensor:
#         bias_exp2 = bias.exp2()
#         scaled_input = input / bias_exp2
#         mantissa, exponent = scaled_input.clamp(-1.0, 1.0).frexp()
#
#         exponent -= (mantissa.abs() <= _SQRT_2_2).to(torch.int)
#         output = mantissa.sgn_().ldexp_(exponent).masked_fill_(exponent <= min_exp, 0) * bias_exp2
#
#         ctx.save_for_backward(input, output, bias)
#         ctx.min_exp = min_exp
#         return output
#
#     @staticmethod
#     def backward(ctx, grad_output: Tensor) -> tuple:
#         scaled_input, output, bias = ctx.saved_tensors
#
#         min_exp = ctx.min_exp
#
#         grad_log2x_sub_b = scaled_input.abs().log2() - bias
#         grad_step = torch.where(
#             scaled_input.abs() >= 1.0,
#             bias.exp2(),
#             torch.where(
#                 grad_log2x_sub_b <= min_exp,
#                 0,
#                 output * (1 + grad_log2x_sub_b.round() - grad_log2x_sub_b)
#             )
#         ) * grad_output * _LOG2
#
#         return grad_output, grad_step, None, None
#
#
# _logarithmic_quant = LogarithmicQuant.apply
#
#
# class _Context(_BaseContext):
#     bias: Tensor
#
#
# class QLogarithmicMethod(QMethod[_Context]):
#     PARAMETERS = (('bias', None),)
#
#     def __init__(self, nbits: int, signed: bool = True, eps: float = 1e-10):
#         assert nbits >= 2
#         assert signed, "Unsigned linear quantization not support yet."
#
#         self.nbits = nbits
#         self.signed = signed
#         exp_bits = nbits - 1 if signed else nbits
#         self.max_int = int(2 ** exp_bits)
#         self.min_exp = 2 - self.max_int
#         self.eps = eps
#
#         self.dtype = ceil_storage_dtype(nbits, signed)
#
#     def init(self, ctx: _Context, shape: tuple[int, ...], dtype, device):
#         ctx.bias = Parameter(torch.empty(shape, dtype=dtype, device=device))
#
#     def __call__(self, input: Tensor, ctx: _Context) -> Tensor:
#         return _logarithmic_quant(input, ctx['bias'], self.min_exp)
#
#     def encode(self, input: Tensor, ctx: _Context) -> Tensor:
#         i = ((input.abs() + self.eps).log2() - ctx.bias).round_().clamp(self.min_exp - 1, 0)
#         if self.signed:
#             return torch.where(input > 0, -i, i - 1).type(self.dtype)
#         else:
#             return (-i).type(self.dtype)
#
#     def decode(self, input: Tensor, ctx: _Context) -> Tensor:
#         bias = ctx['bias']
#         if self.signed:
#             return torch.where(
#                 input >= 0,
#                 (-input.type(bias.dtype) + bias).exp2(),
#                 (input.type(bias.dtype) + 1 + bias).exp2()
#             )
#         else:
#             return (-input.type(bias.dtype) + bias).exp2()
#
#     def _calibrate_saturate_mse(self, input, ctx, dim, bound = (0.2, 1.0), num_step = 80):
#         bias = input.abs().amax(dim=dim, keepdim=True).log2().nan_to_num(0.0)
#
#         best_step = torch.empty_like(bias)
#         best_score = torch.full_like(bias, float('inf'))
#
#         for step_scale in np.linspace(bound[0], bound[1], num_step):
#             step = bias * step_scale
#             input_quant = _logarithmic_quant(input, bias, self.min_exp)
#             score = (input - input_quant).square_().sum(dim=dim, keepdim=True)
#             mask = score < best_score
#
#             best_step[mask] = step[mask]
#             best_score[mask] = score[mask]
#
#         ctx['bias'] = bias
#
#     CALIBRATORS = { 'saturate_mse': _calibrate_saturate_mse }
