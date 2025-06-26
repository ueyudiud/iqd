from diffusers import DiffusionPipeline, UNet2DModel, UNet2DConditionModel
from diffusers.configuration_utils import register_to_config
from torch.nn import Module, Conv2d, Linear

from ..grouping import FullGrouping, ReduceGrouping
from ..method import QSymmetricMethod, QIdentityMethod
from ..mixin import QuantizerMixin
from ..module.conv import Conv2dQ
from ..module.linear import LinearQ
from ..module.mixin import UnaryLinearSetting
from ..module.modeling_utils import transform_submodules
from ..setting import wmap, amap


class PTQ4DMQuantizer(QuantizerMixin):
    @register_to_config
    def __init__(self,
                 wbits: int,
                 abits: int | None,
                 channel_wise: bool = True,
                 calib_num_samples: int = 1024,
                 iters_w: int = 20000,
                 weight: float = 0.01,
                 wwq: bool = False,
                 waq: bool = False,
                 b_start: int = 20,
                 b_end: int = 2,
                 warmup: float = 0.2,
                 lr: float = 4e-5,
                 awq: bool = False,
                 aaq: bool = False,
                 init_wmode: str = "mse",
                 init_amode: str = "mse",
                 order: str = "before",
                 prob: float = 1.0,
                 input_prob: float = 1.0,
                 use_adaround: bool = False,
                 calib_im_mode: str = "random",
                 calib_t_mode: str = "random",
                 calib_t_mode_normal_mean: float = 0.5,
                 calib_t_mode_normal_std: float = 0.35,
                 ):
        super().__init__()

        if wbits is not None:
            weight_method = QSymmetricMethod(wbits)
            weight_calib = { 'mode': init_wmode }
        else:
            weight_method = QIdentityMethod()
            weight_calib = None

        if abits is not None:
            activation_method = QSymmetricMethod(abits)
        else:
            activation_method = QIdentityMethod()

        self.setting = UnaryLinearSetting(
            wmap=wmap(method=weight_method, grouping=ReduceGrouping(0) if channel_wise else FullGrouping(), calibration=weight_calib),
            amap=amap(method=activation_method, grouping=FullGrouping())
        )

    def convert(self, model: UNet2DModel | UNet2DConditionModel, train=True):
        specials = [
            model.time_embedding.linear_1,
            model.time_embedding.linear_2,
            model.conv_out,
        ]

        def _quantize(name: str, module: Module):
            if module in specials:
                return
            elif isinstance(module, Conv2d):
                return Conv2dQ.from_origin(module, self.setting)
            elif isinstance(module, Linear):
                return LinearQ.from_origin(module, self.setting)

        transform_submodules(_quantize, model)
        self._attach_to(model)
        return model


    def quantize(self, pipeline: DiffusionPipeline, **kwargs):
        ...
