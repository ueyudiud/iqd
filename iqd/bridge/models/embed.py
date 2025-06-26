import os
from typing import Optional, Union

import torch
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from torch import Generator, Tensor
from torch.nn import Embedding


class EmbedderMixin(ModelMixin, ConfigMixin):
    def negative_embeds(self, batch_size: int) -> Tensor:
        raise NotImplemented

    def random_embeds(self, batch_size: int, generator: Generator | list[Generator]) -> Tensor:
        raise NotImplemented


class ClassEmbedder(Embedding, EmbedderMixin):
    @register_to_config
    def __init__(self,
                 num_class: int,
                 embedding_dim: int,
                 unclass_index: int | None = None,
                 padding_idx: int = None,
                 max_norm: float = None,
                 norm_type: float = 2.0,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False,
                 _freeze: bool = False,
                 device=None,
                 dtype=None,
    ):
        if unclass_index is not None:
            if unclass_index < 0:
                unclass_index += num_class + 1
            if unclass_index < 0 or unclass_index > num_class:
                raise ValueError("Unconditional feature index out of range.")
        num_embeddings = num_class + 1 if unclass_index is not None else num_class
        super().__init__(num_embeddings, embedding_dim, padding_idx, max_norm, norm_type, scale_grad_by_freq,
                         sparse, None, _freeze, device, dtype)
        self.num_class = num_class
        self.unclass_index = unclass_index

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input)[:, None]

    def _random_embed(self, batch_size: int, generator: Generator) -> Tensor:
        # device on which tensor is created defaults to device
        device = self.weight.device

        labels = torch.randint(0, self.num_class, (batch_size,), dtype=torch.long, generator=generator).to(device=device)

        unclass_index = self.unclass_index
        if unclass_index is not None:
            labels[labels >= unclass_index] += 1

        return labels

    def random_embeds(self, batch_size: int, generator: Generator | list[Generator]) -> Tensor:
        # make sure generator list of length 1 is treated like a non-list
        if isinstance(generator, list) and len(generator) == 1:
            generator = generator[0]

        if isinstance(generator, list):
            latents = [self._random_embed(1, generator[i]) for i in range(batch_size)]
            latents = torch.cat(latents, dim=0)
        else:
            latents = self._random_embed(batch_size, generator)

        return self(latents)

    def negative_embeds(self, batch_size: int) -> Tensor:
        if self.unclass_index is None:
            raise ValueError("Negative class is not available for this model.")
        return self(torch.full((batch_size,), self.unclass_index, device=self.weight.device))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        return super(EmbedderMixin, cls).from_pretrained(pretrained_model_name_or_path, **kwargs)

