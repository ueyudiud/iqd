from diffusers import AutoencoderKL
from diffusers.models.resnet import ResnetBlock2D
from torch.nn import SiLU


class AutoencoderKL2(AutoencoderKL):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: tuple[str, ...] = ("DownEncoderBlock2D",),
        up_block_types: tuple[str, ...] = ("UpDecoderBlock2D",),
        block_out_channels: tuple[int, ...] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        shift_factor: float | None = None,
        latents_mean: tuple[float] | None = None,
        latents_std: tuple[float] | None = None,
        force_upcast: float = True,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        mid_block_add_attention: bool = True,
        legacy: str | None = None
    ):
        super().__init__(in_channels, out_channels, down_block_types, up_block_types, block_out_channels,
                         layers_per_block, act_fn, latent_channels, norm_num_groups, sample_size, scaling_factor,
                         shift_factor, latents_mean, latents_std, force_upcast, use_quant_conv, use_post_quant_conv,
                         mid_block_add_attention)
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