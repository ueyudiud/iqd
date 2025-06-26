import numpy as np
import torch
from diffusers import SchedulerMixin, ConfigMixin, DDIMScheduler
from diffusers.configuration_utils import register_to_config
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from diffusers.utils.torch_utils import randn_tensor

from .scheduling_utils import get_betas


class LegacyDDIMScheduler(SchedulerMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_train_timesteps = 1000,
        beta_start = 0.0001,
        beta_end = 0.02,
        beta_schedule = "linear",
        trained_betas = None,
        clip_sample = True,
        set_alpha_to_one = True,
        steps_offset = 0,
        prediction_type = "epsilon",
        thresholding = False,
        dynamic_thresholding_ratio = 0.995,
        clip_sample_range = 1.0,
        sample_max_value = 1.0,
        timestep_spacing = "leading",
    ):
        betas = get_betas(beta_schedule, trained_betas, num_train_timesteps, beta_start, beta_end)
        alphas = 1. - betas
        alphas_cumprod = alphas.cumprod(dim=0).type(torch.float32)
        self.betas = betas.type(torch.float32)
        self.alphas = alphas.type(torch.float32)
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = (1. - alphas_cumprod) ** 0.5

        # At every step in ddim, we are looking into the previous alphas_cumprod
        # For the final step, there is no previous alphas_cumprod because we are already at 0
        # `set_alpha_to_one` decides whether we set this parameter simply to one or
        # whether we use the final alpha of the "non-previous" one.
        self.final_alpha_cumprod = torch.tensor(1.0, dtype=torch.float32) if set_alpha_to_one else self.alphas_cumprod[0]

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy().astype(np.int64))

    scale_model_input = DDIMScheduler.scale_model_input

    def set_timesteps(self, num_inference_steps, device = None):
        DDIMScheduler.set_timesteps(self, num_inference_steps, device)
        if self.timesteps.amax() < self.config.num_train_timesteps - 1: # Avoid index out of bound.
            self.timesteps = self.timesteps + 1 # Maybe a bug in original LDM project.

    _get_variance = DDIMScheduler._get_variance

    _threshold_sample = DDIMScheduler._threshold_sample

    def step(self, model_output, timestep, sample, eta = 0.0, use_clipped_model_output = False, generator = None, 
             variance_noise = None, return_dict = True):
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        prev_timestep = max(timestep - self.config.num_train_timesteps // self.num_inference_steps, 0)

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep].to(model_output.device)
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep].to(model_output.device) # if prev_timestep >= 0 else self.final_alpha_cumprod
        sqrt_one_minus_alphas_prod_t = self.sqrt_one_minus_alphas_cumprod[timestep].to(model_output.device)
        sqrt_alpha_prod_t = alpha_prod_t.sqrt()

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - sqrt_one_minus_alphas_prod_t * model_output) / sqrt_alpha_prod_t
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - sqrt_alpha_prod_t * pred_original_sample) / sqrt_one_minus_alphas_prod_t
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = sqrt_alpha_prod_t * sample - sqrt_one_minus_alphas_prod_t * model_output
            pred_epsilon = sqrt_alpha_prod_t * model_output + sqrt_one_minus_alphas_prod_t * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        # 4. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (sample - sqrt_alpha_prod_t * pred_original_sample) / sqrt_one_minus_alphas_prod_t

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2).sqrt() * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction

        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                variance_noise = randn_tensor(
                    model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
                )
            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance
        
        if not return_dict:
            return (prev_sample,)

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    def add_noise(self, original_samples, noise, timesteps):
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device=original_samples.device)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod_t = self.alphas_cumprod.gather(-1, timesteps).type(original_samples.dtype) ** 0.5
        sqrt_alpha_prod_t = sqrt_alpha_prod_t.flatten()
        while len(sqrt_alpha_prod_t.shape) < len(original_samples.shape):
            sqrt_alpha_prod_t = sqrt_alpha_prod_t.unsqueeze(-1)

        sqrt_one_minus_alpha_prod_t = self.sqrt_one_minus_alphas_cumprod.gather(-1, timesteps).type(original_samples.dtype)
        sqrt_one_minus_alpha_prod_t = sqrt_one_minus_alpha_prod_t.flatten()
        while len(sqrt_one_minus_alpha_prod_t.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod_t = sqrt_one_minus_alpha_prod_t.unsqueeze(-1)

        noisy_samples = sqrt_alpha_prod_t * original_samples + sqrt_one_minus_alpha_prod_t * noise
        return noisy_samples

    get_velocity = DDIMScheduler.get_velocity

    def __len__(self):
        return self.config.num_train_timesteps