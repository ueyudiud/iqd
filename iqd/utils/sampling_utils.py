from dataclasses import dataclass

from diffusers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from torch import Generator, Tensor

from iqd.bridge.models.embed import EmbedderMixin


@dataclass
class PSampleOutput:
    x_0: Tensor
    eps: Tensor
    alpha_prod_t: Tensor

    @property
    def v_t(self):
        return self.alpha_prod_t ** 0.5 * self.eps - (1 - self.alpha_prod_t) ** 0.5 * self.x_0

    @property
    def snr_t(self):
        return self.alpha_prod_t / (1 - self.alpha_prod_t)


class DDIMSampler:
    scheduler: DDIMScheduler
    embedder: EmbedderMixin | None

    def __init__(self,
                 scheduler: DDIMScheduler,
                 embedder: EmbedderMixin | None = None,
                 ):
        self.scheduler = scheduler
        self.embedder = embedder

    def sample_noise(self, origin: Tensor, generator: Generator | None = None):
        return randn_tensor(origin.shape, generator=generator, device=origin.device)

    def sample_kwargs(self, batch_size: int):
        kwargs = {}

        if self.embedder is not None:
            kwargs['encoder_hidden_states'] = self.embedder.negative_embeds(batch_size).detach()

        return kwargs

    def p_sample(self, x_t: Tensor, t: Tensor, model_output: Tensor) -> PSampleOutput:
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t

        sqrt_alpha_prod_t = alpha_prod_t ** 0.5
        sqrt_beta_prod_t = beta_prod_t ** 0.5

        match self.scheduler.config.prediction_type:
            case 'epsilon':
                x_0_hat = (x_t - sqrt_beta_prod_t * model_output) / sqrt_alpha_prod_t
                eps_hat = model_output
            case 'sample':
                x_0_hat = model_output
                eps_hat = (x_t - sqrt_alpha_prod_t * model_output) / sqrt_beta_prod_t
            case 'v_prediction':
                x_0_hat = sqrt_alpha_prod_t * x_t - sqrt_beta_prod_t * model_output
                eps_hat = sqrt_alpha_prod_t * model_output + sqrt_beta_prod_t * x_t
            case _:
                raise ValueError

        return PSampleOutput(x_0=x_0_hat, eps=eps_hat, alpha_prod_t=alpha_prod_t)

    def q_sample(self, x_0: Tensor, eps: Tensor, t: Tensor):
        alpha_prod_t = self.scheduler.alphas_cumprod[t] if t >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        sqrt_alpha_prod_t = alpha_prod_t ** 0.5
        sqrt_beta_prod_t = beta_prod_t ** 0.5

        return sqrt_alpha_prod_t * x_0 + sqrt_beta_prod_t * eps
