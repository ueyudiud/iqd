import math

import torch
from diffusers.models.embeddings import Timesteps
from torch import Tensor
from torch.nn import Module


def get_timestep_embedding_legacy(
        timesteps,
        embedding_dim,
        flip_sin_to_cos=False,
        downscale_freq_shift=1,
        scale=1,
        max_period=10000,
):
    """
    Legacy version of get_timestep_embedding.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent).to(device=timesteps.device)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb



class LegacyTimesteps(Timesteps):
    def forward(self, timesteps: Tensor) -> Tensor:
        t_emb = get_timestep_embedding_legacy(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


class LegacySiLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)
