from diffusers import VQModel
from diffusers.models.resnet import ResnetBlock2D
from torch.nn import SiLU


class VQModel2(VQModel):
    def __init__(self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: tuple[str, ...] = ("DownEncoderBlock2D",),
        up_block_types: tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: tuple[int, ...] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 3,
        sample_size: int = 32,
        num_vq_embeddings: int = 256,
        norm_num_groups: int = 32,
        vq_embed_dim: int | None = None,
        scaling_factor: float = 0.18215,
        norm_type: str = "group",  # group, spatial
        mid_block_add_attention: bool = True,
        lookup_from_codebook: bool = False,
        force_upcast: bool = False,
        legacy: str | None = None
    ):
        super().__init__(in_channels, out_channels, down_block_types, up_block_types, block_out_channels,
                         layers_per_block, act_fn, latent_channels, sample_size, num_vq_embeddings, norm_num_groups,
                         vq_embed_dim, scaling_factor, norm_type, mid_block_add_attention, lookup_from_codebook,
                         force_upcast)
        self.register_to_config(legacy=legacy)
        self._convert_legacy(legacy)

    def _convert_legacy(self, mode: str | None):
        if mode is not None:
            from diffusers.models.attention_processor import Attention

            from .attn import LegacyVqvaeAttnProcessor
            from .misc import LegacySiLU

            if isinstance(self.encoder.conv_act, SiLU):
                self.encoder.conv_act = LegacySiLU()
            if isinstance(self.decoder.conv_act, SiLU):
                self.decoder.conv_act = LegacySiLU()

            for module in self.modules():
                if isinstance(module, Attention):
                    module.scale = module.inner_dim ** -0.5
                    module.set_processor(LegacyVqvaeAttnProcessor())
                elif isinstance(module, ResnetBlock2D):
                    if isinstance(module.nonlinearity, SiLU):
                        module.nonlinearity = LegacySiLU()