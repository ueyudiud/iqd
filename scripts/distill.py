import argparse
import io
import json
import logging
import os
import sys
from os.path import join as join_path

import torch
from diffusers import DDIMPipeline, DiffusionPipeline, DDIMScheduler, UNet2DModel, UNet2DConditionModel
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from iqd import UNet2DModel2, UNet2DConditionModel2

LOGGER = logging.getLogger(__name__)


def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--device',
        default='cuda',
        help="specify training device"
    )

    parser.add_argument(
        'source',
        help="choose origin model"
    )

    parser.add_argument(
        '--out-dir',
        default='./out/models'
    )

    parser.add_argument(
        '--out-name',
        default=None
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=1
    )

    return parser.parse_args()


def distill(pipeline):
    scheduler: DDIMScheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    origin_unet = pipeline.unet

    device = origin_unet.device

    target_unet_config = dict(origin_unet.config)
    target_unet_config.pop('_use_default_values', None)
    target_unet_config['legacy'] = None
    if isinstance(origin_unet, UNet2DModel):
        target_unet = UNet2DModel2.from_config(target_unet_config).to(device=device)
    elif isinstance(origin_unet, UNet2DConditionModel):
        target_unet = UNet2DConditionModel2.from_config(target_unet_config).to(device=device)
    else:
        raise ValueError(f"Unknown model type {type(origin_unet)}")
    target_unet.load_state_dict(origin_unet.state_dict())

    origin_unet.requires_grad_(False)

    scheduler.set_timesteps(scheduler.config.num_train_timesteps, device=device)

    batch_size = 1
    sample_size = target_unet.config.sample_size
    if isinstance(sample_size, int):
        sample_size = (sample_size, sample_size)
    shape = (batch_size, target_unet.config.out_channels, *sample_size)

    kwargs = {}
    if isinstance(origin_unet, UNet2DConditionModel):
        kwargs['encoder_hidden_states'] = pipeline.embedder.negative_embeds(batch_size).detach()

    optimizer = Adam(target_unet.parameters(), lr=1e-5)
    lr_scheduler = ExponentialLR(optimizer, gamma=0.95)

    num_epoch = 20

    with tqdm(total=num_epoch * scheduler.config.num_train_timesteps) as bar:
        for _ in range(num_epoch):
            x = torch.randn(shape, device=device)
            for t in scheduler.timesteps:
                x = scheduler.scale_model_input(x, t)
                z1 = origin_unet(x, t, **kwargs).sample
                z2 = target_unet(x, t, **kwargs).sample

                l = torch.nn.functional.mse_loss(z1, z2)
                l.backward()
                optimizer.step()
                optimizer.zero_grad()
                del z2

                bar.update(1)
                bar.set_postfix(loss=l.item())

                x = scheduler.step(z1.detach_(), t, x, eta=1.0).prev_sample
                del z1
            lr_scheduler.step()


def main():
    opts = parse_options()

    LOGGER.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    LOGGER.addHandler(console_handler)

    device = opts.device

    torch.manual_seed(29475)

    source_path = opts.source

    LOGGER.info(f"Loading pipeline from {source_path}")

    thin_load = False

    if thin_load:
        pipeline = DDIMPipeline.from_pretrained(source_path)
    else:
        pipeline = DiffusionPipeline.from_pretrained(source_path)

    pipeline.vqvae = None

    LOGGER.info(f"Transferring parameters to {device}")
    pipeline.to(device)

    out_path = join_path(opts.out_dir, opts.out_name or os.path.basename(source_path) + "-distill")

    os.makedirs(out_path, exist_ok=True)

    distill(pipeline)

    if thin_load:
        pipeline = pipeline.to('cpu')
        pipeline = DiffusionPipeline.from_pretrained(source_path, unet=pipeline.unet, scheduler=pipeline.scheduler)

    LOGGER.info('Saving pipeline to %s', out_path)
    pipeline.save_pretrained(out_path)


if __name__ == '__main__':
    main()
