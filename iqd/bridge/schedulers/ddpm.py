import numpy as np
import torch
from diffusers import SchedulerMixin, ConfigMixin, DDPMScheduler
from diffusers.configuration_utils import register_to_config

from .scheduling_utils import get_betas


class LegacyDDPMScheduler(SchedulerMixin, ConfigMixin):
    @register_to_config
    def __init__(
            self,
            num_train_timesteps = 1000,
            beta_start = 0.0001,
            beta_end = 0.02,
            beta_schedule = "linear",
            trained_betas = None,
            variance_type = "fixed_small",
            clip_sample = True,
            prediction_type = "epsilon",
            thresholding = False,
            dynamic_thresholding_ratio = 0.995,
            clip_sample_range = 1.0,
            sample_max_value = 1.0,
            timestep_spacing = "leading",
            steps_offset = 0,
    ):
        betas = get_betas(beta_schedule, trained_betas, num_train_timesteps, beta_start, beta_end)
        self.betas = betas.to(torch.float32)
        self.alphas = (1. - self.betas).to(torch.float32)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(torch.float32)
        self.one = torch.tensor(1.0)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.custom_timesteps = False
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))

        self.variance_type = variance_type

    scale_model_input = DDPMScheduler.scale_model_input

    set_timesteps = DDPMScheduler.set_timesteps

    _get_variance = DDPMScheduler._get_variance

    _threshold_sample = DDPMScheduler._threshold_sample

    step = DDPMScheduler.step

    add_noise = DDPMScheduler.add_noise

    get_velocity = DDPMScheduler.get_velocity

    def __len__(self):
        return self.config.num_train_timesteps