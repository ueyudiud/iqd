from diffusers import UNet2DConditionModel
from diffusers.models.embeddings import Timesteps


class UNet2DConditionModel2(UNet2DConditionModel):
    def __init__(
        self,
        sample_size: int | None = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: tuple[str, ...] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: str | None = "UNetMidBlock2DCrossAttn",
        up_block_types: tuple[str, ...] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        only_cross_attention: bool | tuple[bool, ...] = False,
        block_out_channels: tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int | tuple[int, ...] = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        dropout: float = 0.0,
        act_fn: str = "silu",
        norm_num_groups: int | None = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int | tuple[int, ...] = 1280,
        transformer_layers_per_block: int | tuple[int, ...] | tuple[tuple[int, ...]] = 1,
        reverse_transformer_layers_per_block: tuple[tuple[int]] | None = None,
        encoder_hid_dim: int | None = None,
        encoder_hid_dim_type: str | None = None,
        attention_head_dim: int | tuple[int, ...] | None = 8,
        num_attention_heads: int | tuple[int, ...] | None = None,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: str | None = None,
        addition_embed_type: str | None = None,
        addition_time_embed_dim: int | None = None,
        num_class_embeds: int | None = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        resnet_skip_time_act: bool = False,
        resnet_out_scale_factor: float = 1.0,
        time_embedding_type: str = "positional",
        time_embedding_dim: int | None = None,
        time_embedding_act_fn: str | None = None,
        timestep_post_act: str | None = None,
        time_cond_proj_dim: int | None = None,
        conv_in_kernel: int = 3,
        conv_out_kernel: int = 3,
        projection_class_embeddings_input_dim: int | None = None,
        attention_type: str = "default",
        class_embeddings_concat: bool = False,
        mid_block_only_cross_attention: bool | None = None,
        cross_attention_norm: str | None = None,
        addition_embed_type_num_heads: int = 64,
        legacy: str | None = None,
    ):
        super().__init__(sample_size, in_channels, out_channels, center_input_sample, flip_sin_to_cos, freq_shift,
                         down_block_types, mid_block_type, up_block_types, only_cross_attention, block_out_channels,
                         layers_per_block, downsample_padding, mid_block_scale_factor, dropout, act_fn, norm_num_groups,
                         norm_eps, cross_attention_dim, transformer_layers_per_block,
                         reverse_transformer_layers_per_block, encoder_hid_dim, encoder_hid_dim_type,
                         attention_head_dim, num_attention_heads, dual_cross_attention, use_linear_projection,
                         class_embed_type, addition_embed_type, addition_time_embed_dim, num_class_embeds,
                         upcast_attention, resnet_time_scale_shift, resnet_skip_time_act, resnet_out_scale_factor,
                         time_embedding_type, time_embedding_dim, time_embedding_act_fn, timestep_post_act,
                         time_cond_proj_dim, conv_in_kernel, conv_out_kernel, projection_class_embeddings_input_dim,
                         attention_type, class_embeddings_concat, mid_block_only_cross_attention, cross_attention_norm,
                         addition_embed_type_num_heads)
        self.register_to_config(legacy=legacy)
        self._convert_legacy(legacy)

    def _convert_legacy(self, mode: str | None):
        if mode is not None:
            from diffusers.models.attention_processor import Attention, AttnProcessor, AttnProcessor2_0

            from .attn import LegacyCrossAttnProcessor
            from .misc import LegacyTimesteps

            time_proj = self.time_proj
            if isinstance(time_proj, Timesteps):
                self.time_proj = LegacyTimesteps(
                    time_proj.num_channels,
                    time_proj.flip_sin_to_cos,
                    time_proj.downscale_freq_shift,
                    time_proj.scale
                )
            for module in self.modules():
                if isinstance(module, Attention):
                    processor = module.processor
                    if isinstance(processor, AttnProcessor2_0):
                        module.set_processor(LegacyCrossAttnProcessor())
                    elif isinstance(processor, AttnProcessor):
                        raise NotImplemented