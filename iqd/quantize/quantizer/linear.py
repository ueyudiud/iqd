import logging
from dataclasses import dataclass
from typing import Sized

import torch
from diffusers import DiffusionPipeline, UNet2DModel, DDIMScheduler, UNet2DConditionModel
from diffusers.configuration_utils import register_to_config
from torch import Generator, Tensor
from torch.nn import Module
from torch.nn.modules import Conv2d, Linear
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from iqd.utils.sampling_utils import DDIMSampler
from ..grouping import FullGrouping, TimedGrouping, CatGrouping
from ..method import QIdentityMethod, QLinearMethod
from ..method.base import QFunction
from ..mixin import QuantizerMixin
from ..model.unet import QUNet2DModel, QUNet2DConditionModel
from ..module.conv import Conv2dQ
from ..module.leaf import MaskedLeafQ, LeafQ
from ..module.linear import LinearQ
from ..module.mixin import UnaryLinearSetting, UnaryLinearQ
from ..module.modeling_utils import transform_submodules


@dataclass
class PruningState:
    quantized_weight_rate: float
    quantized_activation_rate: float


class Target:
    weights: list[tuple[Parameter, MaskedLeafQ]]
    activations: list[QFunction]
    parameters: set[Parameter]

    def __init__(self, origin_unet: UNet2DModel, target_unet: QUNet2DModel):
        weights = []
        activations = []
        parameters = set(target_unet.parameters())
        num_param = 0
        for name, module in target_unet.named_modules():
            if name.endswith('.w_layer'):
                param = origin_unet.get_parameter(name.replace('.w_layer', '.weight'))
                weights.append((param, module))
                for remove_parameter in module.function.parameters():
                    remove_parameter.requires_grad_(False)
                    parameters.remove(remove_parameter)
                num_param += param.numel()
            elif name.endswith('.a_layer'):
                activations.append(module)

        self.activations = activations
        self.weights = weights
        self.parameters = parameters
        self.num_param = num_param

    def diff_loss(self):
        loss = 0
        for origin, module in self.weights:
            loss += (origin - module()).abs().sum()
        loss /= self.num_param
        return loss


class LinearQuantizer(QuantizerMixin):
    num_timestep: int | None
    scheduler: DDIMScheduler | None
    sampler: DDIMSampler | None

    @register_to_config
    def __init__(self,
                 wbits: int,
                 abits: int,
                 num_timestep: int | None = None,
                 *,
                 lambda_r: float = 0.0,
                 init_wmode: str = 'saturate',
                 init_amode: str = 'ema-saturate',
                 split_shortcut: bool = False,
                 weight_schedule: str = '1',
                 ):
        assert wbits == 32 or 2 <= wbits <= 8, ValueError(f"Unsupported weight bits: {wbits}")
        assert abits == 32 or 2 <= abits <= 8, ValueError(f"Unsupported activation bits: {abits}")
        assert init_wmode in ('minmax', 'saturate'), ValueError(f"Unsupported weight initialize mode: {init_wmode}")
        assert init_amode in ('ema-minmax', 'ema-saturate'), ValueError(f"Unsupported weight initialize mode: {init_amode}")
        assert weight_schedule in ('1', 'snr', 'max-snr-1', 'min-snr-5'), ValueError(f"Unsupported weight schedule: {weight_schedule}")

        super().__init__()

        self.num_timestep = num_timestep
        self.sampler = None
        self.scheduler = None

        if wbits == 32:
            weight_type = QIdentityMethod()
            weight_calibration = None
        else:
            weight_type = QLinearMethod(wbits)
            weight_calibration = { 'mode': init_wmode }

        if abits == 32:
            activation_type = QIdentityMethod()
            activation_calibration = None
        else:
            activation_type = QLinearMethod(abits)
            match init_amode:
                case 'ema-minmax':
                    activation_calibration = { 'mode': init_amode, 'center': 'min', 'step_scale': 0.4 }
                case 'ema-saturate':
                    activation_calibration = { 'mode': init_amode, 'num_step_candidate': 40 }

        wmap = self._wmap
        amap = self._amap

        self._weight_type = weight_type
        self._activation_type = activation_type
        self._weight_calibration = weight_calibration
        self._activation_calibration = activation_calibration
        self.time_emb_setting = UnaryLinearSetting(
            wmap=wmap(weight_type, FullGrouping(), weight_calibration),
            amap=amap(activation_type, FullGrouping(), activation_calibration),
        )
        self.linear_setting = UnaryLinearSetting(
            wmap=wmap(weight_type, FullGrouping(), weight_calibration),
            amap=amap(activation_type, TimedGrouping(FullGrouping(), num_timestep), activation_calibration),
        )
        self.conv2d_setting = UnaryLinearSetting(
            wmap=wmap(weight_type, FullGrouping(), weight_calibration),
            amap=amap(activation_type, TimedGrouping(FullGrouping(), num_timestep), activation_calibration),
        )
        self.lambda_r = lambda_r
        self.ema_slot = activation_calibration or {}
        self._train = True

    def set_schedule(self, pipeline, device):
        scheduler = pipeline.scheduler
        scheduler = scheduler.from_config(scheduler.config)  # Clone a scheduler

        self.sampler = DDIMSampler(scheduler, getattr(pipeline, 'embedder', None))
        self.scheduler = scheduler
        self.num_timestep = self.config.num_timestep if self.config.num_timestep is not None else scheduler.config.num_train_timesteps
        self.register_to_config(num_timestep=self.num_timestep)

        scheduler.set_timesteps(self.num_timestep, device)

    def _wmap(self, method, grouping, calibration):
        def apply(value: Tensor, /):
            function = method(grouping(*value.shape), value.dtype, value.device)
            function.calibration = calibration
            return LeafQ(function, value.shape, dtype=value.dtype, device=value.device)

        return apply

    def _amap(self, method, grouping, calibration):
        def apply(shape, dtype, device):
            function = method(grouping(*shape), dtype, device)
            function.calibration = calibration
            return function

        return apply

    def convert_origin(self, model: UNet2DModel | UNet2DConditionModel):
        config = dict(model.config)
        config.pop('_use_default_values', None)
        if isinstance(model, UNet2DModel):
            qmodel = QUNet2DModel.from_config(config, _load_origin=True)
            qmodel.load_state_dict(model.state_dict())
            return self.convert(qmodel, train=True)
        elif isinstance(model, UNet2DConditionModel):
            qmodel = QUNet2DConditionModel.from_config(config, _load_origin=True)
            qmodel.load_state_dict(model.state_dict())
            return self.convert(qmodel, train=True)
        else:
            raise ValueError(f"Unsupported model: {type(model)}")

    def convert(self, model: QUNet2DModel | QUNet2DConditionModel, train=False):
        from iqd.utils import globals

        self._train = train

        specials = [
            model.time_embedding.linear_1,
            model.time_embedding.linear_2,
            model.conv_in,
            model.conv_out,
        ]

        # Skip sampler.
        for down_block in model.down_blocks:
            if down_block.downsamplers is not None:
                for downsampler in down_block.downsamplers:
                    specials.append(downsampler)

        # Cut shortcut layer
        if self.config.split_shortcut:
            num_chan_input = model.config.block_out_channels[-1]
            for up_block in model.up_blocks:
                for resnet in up_block.resnets:
                    module = resnet.conv1
                    num_chan_output = module.weight.size(0)
                    split_sizes = [num_chan_input, module.weight.size(1) - num_chan_input]
                    module = Conv2dQ.from_origin(module, UnaryLinearSetting(
                        wmap=self._wmap(self._weight_type, CatGrouping(split_sizes, 1), self._weight_calibration),
                        amap=self._amap(self._activation_type, TimedGrouping(CatGrouping(split_sizes, 1), self.num_timestep), self._activation_calibration),
                    ))
                    resnet.conv1 = module
                    specials.append(module)
                    num_chan_input = num_chan_output

        def _quantize(name: str, module: Module):
            if module in specials:
                return
            elif isinstance(module, Conv2d):
                if 'shortcut' in name or 'downsamplers' in name:
                    return
                elif 'conv1' in name and 'up_blocks' in name:
                    return Conv2dQ.from_origin(module, self.shortcut_conv2d_setting)
                else:
                    return Conv2dQ.from_origin(module, self.conv2d_setting)
            elif isinstance(module, Linear):
                if 'time_emb_proj' in name:
                    return LinearQ.from_origin(module, self.time_emb_setting)
                else:
                    return LinearQ.from_origin(module, self.linear_setting)

        try:
            globals.total_timestep = 1000
            transform_submodules(_quantize, model)
        finally:
            globals.total_timestep = None

        self._attach_to(model)
        return model

    @torch.no_grad()
    def _calibrate(self,
                   target: Target,
                   unet: QUNet2DModel,
                   samples: Dataset[Tensor],
                   batch_size: int,
                   generator: Generator,
                   ):
        from iqd.utils import globals

        if self.config.wbits != 32:
            for origin_weight, target_module in target.weights:
                target_module.function.calibrate(origin_weight)

        if self.config.abits == 32:
            return

        def _calibrate(module: UnaryLinearQ, input: tuple):
            input, = input
            module.a_layer.calibrate(input)

        sample_loader = DataLoader(samples, batch_size)

        progress = tqdm(total=len(sample_loader) * len(self.scheduler.timesteps), unit="batch")

        kwargs = self.sampler.sample_kwargs(batch_size)

        handles = []
        try:
            for name, module in unet.named_modules():
                if isinstance(module, UnaryLinearQ):
                    handles.append(module.register_forward_pre_hook(_calibrate))

            ema_step = 0
            for batched_samples in sample_loader:
                self.ema_slot['ema_step'] = ema_step
                batched_samples = batched_samples.to(unet.device)

                for timestep in self.scheduler.timesteps:
                    batched_epsilon = self.sampler.sample_noise(batched_samples, generator)
                    batched_noised_samples = self.sampler.q_sample(batched_samples, batched_epsilon, timestep)

                    globals.current_timestep = timestep
                    globals.stored_timestep = timestep // (self.scheduler.config.num_train_timesteps // self.num_timestep)
                    _ = unet(batched_noised_samples, timestep, **kwargs)

                    del batched_noised_samples
                    del batched_epsilon

                    progress.update()

                del batched_samples

                ema_step += 1

        finally:
            globals.current_timestep = None
            globals.stored_timestep = None
            for handle in handles:
                handle.remove()

    def quantize(self,
                 pipeline: DiffusionPipeline,
                 calib_samples: Dataset[Tensor] | Sized,
                 batch_size: int,
                 generator: Generator | None = None,
                 device = None,
                 logger: logging.Logger | None = None,
                 **kwargs):
        if device is None:
            device = pipeline.device
        else:
            pipeline.to(device=device)

        # Step1: Prepare quantized model.
        logger.info("Prepare quantized model.")
        origin_unet: UNet2DModel | UNet2DConditionModel = pipeline.unet

        target_unet = self.convert_origin(origin_unet).to(device=device)
        pipeline.unet = target_unet
        self.set_schedule(pipeline, device)

        origin_unet.requires_grad_(False)

        # Step2: Calibrate quantize parameters.
        logger.info("Calibrating model.")
        target = Target(origin_unet, target_unet)
        self._calibrate(target, target_unet, calib_samples, batch_size, generator=generator)

        profile = { }

        # Finish quantization.
        pipeline.unet = target_unet.detach_()

        return pipeline, profile
