from typing import Callable

import torch
from torch import Tensor
from torch.nn import Module


def empty_tensor(shape, dtype, device, out: Tensor | None):
    if not isinstance(shape, tuple):
        shape = (shape,)

    if out is not None:
        assert (
                out.shape == shape and
                out.dtype == dtype and
                out.device == device
        ), ValueError("Tensor and metadata not match.")
        return out

    return torch.empty(shape, dtype=dtype, device=device)


def transform_submodules(transformer: Callable[[str, Module], Module | None], module: Module) -> Module:
    submodules = module._modules
    for name, child in submodules.items():
        mapped_child = transformer(name, child)
        if mapped_child is not None:
            submodules[name] = mapped_child
        else:
            transform_submodules(transformer, child)
    return module
