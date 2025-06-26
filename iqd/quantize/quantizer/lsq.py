import logging
from dataclasses import dataclass
from typing import Any, Callable, Sized

import torch
from diffusers import UNet2DModel, UNet2DConditionModel, DiffusionPipeline, DDIMScheduler
from diffusers.configuration_utils import register_to_config
from torch import Tensor, Generator
from torch.nn import Module, Conv2d, Linear
from torch.nn.parameter import Parameter
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from iqd.utils.sampling_utils import DDIMSampler
from ..grouping import FullGrouping, TimedGrouping
from ..method import QIdentityMethod, QLinearMethod
from ..method.base import QFunction
from ..mixin import QuantizerMixin
from ..model import QUNet2DModel, QUNet2DConditionModel
from ..module import Conv2dQ, LinearQ, PseudoLeafQ, LeafQ
from ..module.mixin import UnaryLinearSetting, UnaryLinearQ
from ..module.modeling_utils import transform_submodules


@dataclass
class PruningState:
    quantized_weight_rate: float
    quantized_activation_rate: float


class Target:
    weights: list[tuple[Parameter, PseudoLeafQ]]
    activations: list[QFunction]

    def __init__(self, origin_unet: UNet2DModel, target_unet: QUNet2DModel):
        weights = []
        activations = []
        num_param = 0
        for name, module in target_unet.named_modules():
            if name.endswith('.w_layer'):
                param = origin_unet.get_parameter(name.replace('.w_layer', '.weight'))
                weights.append((param, module))
                num_param += param.numel()
                module.requires_grad_(False)
            elif name.endswith('.a_layer'):
                activations.append(module)

        self.activations = activations
        self.weights = weights
        self.num_param = num_param

    def diff_loss(self):
        loss = 0
        for origin, module in self.weights:
            loss += (origin - module()).abs().sum()
        loss /= self.num_param
        return loss


@dataclass
class Stage:
    epoch: int = 1
    apr: float = 1.0
    wpr: float = 1.0
    lr: float | None = None
    lr_decay: float | None = None


class LSQQuantizer(QuantizerMixin):
    num_timestep: int | None
    scheduler: DDIMScheduler | None
    sampler: DDIMSampler | None
    stages: list[Stage]

    @register_to_config
    def __init__(self,
                 wbits: int,
                 abits: int,
                 num_timestep: int | None = None,
                 *,
                 lambda_r: float = 0.0,
                 init_wmode: str = 'saturate',
                 init_amode: str = 'ema-saturate',
                 lr: float = 1e-5,
                 lr_decay: float = 0.9,
                 num_epoch: int = 5,
                 weight_schedule: str = '1'
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

    def set_schedule(self, scheduler: DDIMScheduler):
        scheduler = scheduler.from_config(scheduler.config)  # Clone a scheduler

        self.sampler = DDIMSampler(scheduler)
        self.scheduler = scheduler
        self.num_timestep = self.config.num_timestep if self.config.num_timestep is not None else scheduler.config.num_train_timesteps
        self.register_to_config(num_timestep=self.num_timestep)

        scheduler.set_timesteps(self.num_timestep)

    def _wmap(self, method, grouping, calibration):
        def apply(value: Tensor, /):
            function = method(grouping(*value.shape), value.dtype, value.device)
            function.calibration = calibration
            if self._train:
                return PseudoLeafQ(function, value)
            else:
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

        for module in model.down_blocks:
            if module.downsamplers is not None:
                for downsampler in module.downsamplers:
                    specials.append(downsampler)

        def _quantize(name: str, module: Module):
            if module in specials:
                return
            elif isinstance(module, Conv2d):
                if 'shortcut' in name or 'downsamplers' in name:
                    return
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
                    _ = unet(batched_noised_samples, timestep)

                    progress.update()

                ema_step += 1

        finally:
            globals.current_timestep = None
            globals.stored_timestep = None
            for handle in handles:
                handle.remove()

    def _quantize_aware_train(self,
                              target_unet: QUNet2DModel,
                              origin_unet: UNet2DModel,
                              samples: Dataset[Tensor],
                              batch_size: int,
                              optimizer: Optimizer,
                              target: Target,
                              generator: Generator | None,
                              progress: tqdm,
                              state: dict[str, Any]) -> dict:
        total_loss = []
        total_loss_r = []
        total_timesteps = []

        data_loader = DataLoader(samples, batch_size, shuffle=True)

        kwargs = self.sampler.sample_kwargs(batch_size)

        for batched_samples in data_loader:
            for timestep in self.scheduler.timesteps[torch.randperm(self.num_timestep, generator=generator)]:
                batched_epsilon = self.sampler.sample_noise(batched_samples, generator)
                batched_noised_samples = self.sampler.q_sample(batched_samples, batched_epsilon, timestep)
                alpha_cum_prod = self.scheduler.alphas_cumprod[timestep]
                snr = alpha_cum_prod / (1 - alpha_cum_prod)

                origin_model_output = origin_unet(batched_noised_samples, timestep, **kwargs).sample
                target_model_output = target_unet(batched_noised_samples, timestep, **kwargs).sample

                match self.config.weight_schedule:
                    case '1':
                        loss_scale = 1
                    case 'snr':
                        loss_scale = snr
                    case 'max-snr-1':
                        loss_scale = max(snr, 1)
                    case 'min-snr-5':
                        loss_scale = min(snr, 5)
                    case _:
                        raise ValueError

                loss = torch.nn.functional.mse_loss(origin_model_output, target_model_output) * loss_scale
                loss_r = target.diff_loss()

                loss.add_(loss_r, alpha=self.lambda_r)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss.append(loss.item())
                total_loss_r.append(loss_r.item())
                total_timesteps.append(timestep.item())

                progress.update()

                state.update(t=timestep.item(), loss=loss.item(), loss_r=loss_r.item())
                progress.set_postfix(state)

        return {
            'loss': total_loss,
            'loss_r': total_loss_r,
            'timesteps': total_timesteps,
        }

    def quantize(self,
                 pipeline: DiffusionPipeline,
                 train_samples: Dataset[Tensor] | Sized,
                 calib_samples: Dataset[Tensor] | Sized,
                 batch_size: int,
                 callback: Callable[..., None] | None = None,
                 generator: Generator | None = None,
                 device = None,
                 logger: logging.Logger | None = None):
        if device is None:
            device = pipeline.device
        else:
            pipeline.to(device=device)

        # Step1: Prepare quantized model.
        logger.info("Prepare quantized model.")
        origin_unet: UNet2DModel | UNet2DConditionModel = pipeline.unet

        target_unet = self.convert_origin(origin_unet).to(device=device)
        pipeline.unet = target_unet
        self.set_schedule(pipeline.scheduler)

        origin_unet.requires_grad_(False)

        # Step2: Calibrate quantize parameters.
        logger.info("Calibrating model.")
        target = Target(origin_unet, target_unet)
        self._calibrate(target, target_unet, calib_samples, batch_size, generator=generator)

        # Step3: Preform QAT.
        total_epoch = sum(stage.epoch for stage in self.stages)

        optimizer = torch.optim.Adam(target_unet.parameters(), lr=self.config.lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.config.lr_decay)
        progress_scale = ((len(train_samples) - 1) // batch_size + 1) * self.num_timestep

        num_stage = 0
        num_step = 0

        loss_list = []
        loss_r_list = []
        timesteps_list = []

        profile = {
            'total_epoch': total_epoch,
            'train_loss': loss_list,
            'train_loss_r': loss_r_list,
            'timesteps': timesteps_list
        }

        if callback is not None:
            callback(pipeline, num_stage=num_stage, num_step=num_step)

        logger.info("Quantized training model.")
        with tqdm(total=progress_scale * total_epoch, unit='iteration') as progress:
            for epoch in range(self.config.num_epoch):
                state = {}
                epoch_profile = self._quantize_aware_train(
                    target_unet,
                    origin_unet,
                    train_samples,
                    batch_size=batch_size,
                    optimizer=optimizer,
                    target=target,
                    generator=generator,
                    progress=progress,
                    state=state,
                )
                loss_list.extend(epoch_profile['loss'])
                loss_r_list.extend(epoch_profile['loss_r'])
                timesteps_list.extend(epoch_profile['timesteps'])

                num_step += len(train_samples)

                if callback is not None:
                    callback(pipeline, num_stage=num_stage, num_step=num_step)

                lr_scheduler.step()
                epoch += 1

        # Finish quantization.
        pipeline.unet = target_unet.detach_()

        return pipeline, profile
