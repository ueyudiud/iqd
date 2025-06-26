from typing import Any

import torch
from torch.linalg import Tensor

from iqd.bridge.models.unet import UNet2DModel2
from iqd.bridge.models.unet_cond import UNet2DConditionModel2
from ..mixin import QModelMixin


class QUNet2DModel(UNet2DModel2, QModelMixin):
    def forward(self, sample: Tensor, timestep: int | Tensor, class_labels=None, return_dict=True):
        if not isinstance(timestep, Tensor):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        from iqd.utils import globals
        try:
            globals.total_timestep = 1000 # TODO
            globals.current_timestep = timestep
            return super().forward(sample, timestep, class_labels, return_dict)
        finally:
            globals.current_timestep = None
            globals.total_timestep = None

    @property
    def dtype(self):
        return torch.float


class QUNet2DConditionModel(UNet2DConditionModel2, QModelMixin):
    def forward(
        self,
        sample: Tensor,
        timestep: Tensor | float | int,
        encoder_hidden_states: Tensor,
        class_labels: Tensor | None = None,
        timestep_cond: Tensor | None = None,
        attention_mask: Tensor | None = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
        added_cond_kwargs: dict[str, Tensor] | None = None,
        down_block_additional_residuals: tuple[Tensor] | None = None,
        mid_block_additional_residual: tuple[Tensor] | None = None,
        down_intrablock_additional_residuals: tuple[Tensor] | None = None,
        encoder_attention_mask: Tensor | None = None,
        return_dict: bool = True,
    ):
        if not isinstance(timestep, Tensor):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        from iqd.utils import globals
        try:
            globals.total_timestep = 1000 # TODO
            globals.current_timestep = timestep
            return super().forward(sample, timestep, encoder_hidden_states, class_labels, timestep_cond, attention_mask,
                                   cross_attention_kwargs, added_cond_kwargs, down_block_additional_residuals,
                                   mid_block_additional_residual, down_intrablock_additional_residuals,
                                   encoder_attention_mask, return_dict)
        finally:
            globals.current_timestep = None
            globals.total_timestep = None

    @property
    def dtype(self):
        return torch.float