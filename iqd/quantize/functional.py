from abc import ABC

import torch.autograd
from torch import Tensor
from torch.autograd import Function


class LsqContext(ABC, torch.autograd.function.FunctionCtx):
    lower: float
    upper: float
    saved_tensors: tuple[Tensor]


class Lsq2iContext(ABC, torch.autograd.function.FunctionCtx):
    lower: float
    upper: float
    saved_tensors: tuple[Tensor]


class Lsq(Function):
    @staticmethod
    def forward(ctx: LsqContext, input: Tensor, step: Tensor, lower: float, upper: float):
        input = input / step
        ctx.save_for_backward(input)
        ctx.lower = lower
        ctx.upper = upper
        input = input.clamp(lower, upper)
        input = input.round_()
        input *= step
        return input

    @staticmethod
    def backward(ctx: LsqContext, grad_output: Tensor):
        input, = ctx.saved_tensors
        lower = ctx.lower
        upper = ctx.upper

        grad_input = grad_output

        grad_step = input.round().sub_(input)
        grad_step.masked_fill_(input <= lower, lower)
        grad_step.masked_fill_(input >= upper, upper)
        grad_step *= grad_output

        return grad_input, grad_step, None, None


class Lsq2i(Function):
    @staticmethod
    def forward(ctx: Lsq2iContext, input: Tensor, step: Tensor, zero_point: Tensor, lower: int, upper: int):
        input = input / step - zero_point
        ctx.save_for_backward(input)
        ctx.lower = lower
        ctx.upper = upper
        input = input.clamp(lower, upper).round_()
        input += zero_point
        input *= step
        return input

    @staticmethod
    def backward(ctx: Lsq2iContext, grad_output: Tensor):
        input, = ctx.saved_tensors
        lower = ctx.lower
        upper = ctx.upper
        lower_mask = input <= lower
        upper_mask = input >= upper

        grad_input = grad_output

        grad_step = input.round().sub_(input)
        grad_step = torch.where(lower_mask, lower, torch.where(upper_mask, upper,grad_step))
        grad_step *= grad_output

        return grad_input, grad_step, None, None, None


class Lsq2(Function):
    @staticmethod
    def forward(ctx: LsqContext, input: Tensor, step: Tensor, bias: Tensor, lower: float, upper: float):
        input = (input - bias) / step
        ctx.save_for_backward(input)
        ctx.lower = lower
        ctx.upper = upper
        input = input.clamp(lower, upper)
        input = input.round_()
        input *= step
        input += bias
        return input

    @staticmethod
    def backward(ctx: LsqContext, grad_output: Tensor):
        input, = ctx.saved_tensors
        lower = ctx.lower
        upper = ctx.upper
        lower_mask = input <= lower
        upper_mask = input >= upper

        grad_input = grad_output

        grad_step = input.round().sub_(input)
        grad_step.masked_fill_(lower_mask, lower)
        grad_step.masked_fill_(upper_mask, upper)
        grad_step *= grad_output

        grad_bias = grad_output.clone()
        grad_bias.masked_fill_(lower_mask, 0)
        grad_bias.masked_fill_(upper_mask, 0)

        return grad_input, grad_step, grad_bias, None, None

lsq = Lsq.apply
lsq2i = Lsq2i.apply
lsq2 = Lsq2.apply