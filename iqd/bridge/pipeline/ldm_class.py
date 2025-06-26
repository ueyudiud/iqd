import inspect

import torch
from diffusers import DiffusionPipeline, ImagePipelineOutput
from diffusers.models import AutoencoderKL, UNet2DConditionModel, UNet2DModel, VQModel
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils.torch_utils import randn_tensor
from torch import Tensor, Generator

from ..models.embed import EmbedderMixin


class LDMClassedPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "embedder->unet->vqvae"

    vqvae: VQModel | AutoencoderKL
    unet: UNet2DModel | UNet2DConditionModel
    embedder: EmbedderMixin
    scheduler: DDIMScheduler | PNDMScheduler | LMSDiscreteScheduler

    def __init__(
        self,
        vqvae: VQModel | AutoencoderKL,
        unet: UNet2DModel | UNet2DConditionModel,
        embedder: EmbedderMixin,
        scheduler: DDIMScheduler | PNDMScheduler | LMSDiscreteScheduler,
    ):
        super().__init__()
        self.register_modules(vqvae=vqvae, unet=unet, embedder=embedder, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int | None = None,
        class_labels: Tensor | None = None,
        num_inference_steps: int | None = 50,
        guidance_scale: float = 1.0,
        eta: float | None = 0.0,
        generator: Generator | list[Generator] | None = None,
        latents: Tensor | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        skip_decode: bool = False,
        **kwargs,
    ) -> tuple | ImagePipelineOutput:
        if batch_size is None:
            if class_labels is not None:
                batch_size = len(class_labels)
            elif latents is not None:
                batch_size = len(latents)
            else:
                batch_size = 1
        elif isinstance(batch_size, Tensor) and class_labels is None:
            class_labels = batch_size
            batch_size = len(batch_size)

        if class_labels is None:
            class_embeds = self.embedder.random_embeds(batch_size, generator)
        else:
            class_embeds = self.embedder(class_labels)
            if len(class_labels) != batch_size and len(class_labels) != 1:
                raise ValueError(
                    f"Batch size ({batch_size}) does not match context size ({len(class_labels)})"
                )

        if latents is None:
            latents = randn_tensor(
                    (batch_size, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size),
                    generator=generator)
        elif len(latents) != batch_size:
            raise ValueError(
                f"Batch size ({batch_size}) does not match latents size ({len(latents)})"
            )
        latents = latents.to(self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())

        extra_kwargs = {}
        if accepts_eta:
            extra_kwargs["eta"] = eta

        if guidance_scale != 1.0:
            negative_class_embeds = self.embedder.negative_embeds(batch_size)

        for t in self.progress_bar(self.scheduler.timesteps):
            if guidance_scale == 1.0:
                latent_input = latents
                context = class_embeds
            else:
                latent_input = torch.cat([latents, latents])
                context = torch.cat([negative_class_embeds, class_embeds])
            # predict the noise residual
            noise_pred = self.unet(latent_input, t, encoder_hidden_states=context).sample

            if guidance_scale != 1.0:
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_kwargs).prev_sample

        # scale and decode the image latents with vae
        if skip_decode:
            image = latents
        else:
            latents = 1 / self.vqvae.config.scaling_factor * latents
            image = self.vqvae.decode(latents).sample
        
        if output_type == "pt":
            image = image.cpu()
        else:
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            if output_type == "pil":
                image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


