import argparse
import inspect
import json
import logging
import os
import sys
import time
from typing import Generator

import numpy as np
import torch
from diffusers import DiffusionPipeline, DDPMPipeline, DDIMPipeline, LDMPipeline
from tqdm import tqdm

from iqd import LDMClassedPipeline

LOGGER = logging.getLogger(__name__)


def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--device', 
        default='cuda'
    )

    parser.add_argument(
        'model'
    )

    parser.add_argument(
        '-o', '--output',
        default='./out/?'
    )

    parser.add_argument(
        '-f', '--format',
        default='jpg',
        choices=['jpg', 'png', 'npy', 'npz']
    )

    dump_config_group = parser.add_mutually_exclusive_group()

    dump_config_group.add_argument(
        '--dump-config',
        action='store_true',
    )

    dump_config_group.add_argument(
        '--no-dump-config',
        action='store_false',
        dest='dump_config',
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=1
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=1
    )

    parser.add_argument(
        '--num-infer-step',
        type=int,
        default=100
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None
    )

    parser.add_argument(
        '--eta',
        type=float,
        default=0.0
    )

    parser.add_argument(
        '--guidance-scale',
        type=float,
        default=1.0
    )

    parser.add_argument(
        '--solver',
        choices=['any', 'ddim', 'ddim-c'],
        default='any'

    )

    parser.add_argument(
        '--condition',
        choices=['uniform', 'random'],
        default='random'
    )

    return parser.parse_args()

def split_batch(total_size: int, batch_size: int):
    index = 0
    while index + batch_size < total_size:
        yield index, batch_size
        index += batch_size
    yield index, total_size - index

def sample(pipeline: DDPMPipeline | DDIMPipeline | LDMPipeline | LDMClassedPipeline, **kwargs) -> Generator:
    seed = kwargs['seed']
    format = kwargs['format']
    sample_size = kwargs['sample_size']
    batch_size = kwargs['batch_size']
    solver = kwargs['solver']

    if format in ('npy', 'npz'):
        pipeline_output_type = 'np'
        index_wrapper = slice
    else:
        pipeline_output_type = 'pil'
        index_wrapper = range

    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    pipeline_kwargs = {
        'num_inference_steps': kwargs['num_infer_step'],
        'generator': generator,
        'output_type': pipeline_output_type,
        'return_dict': False
    }
    pipeline_param_keys = inspect.signature(pipeline.__call__).parameters.keys()
    if 'eta' in pipeline_param_keys:
        pipeline_kwargs['eta'] = kwargs['eta']
    if 'guidance_scale' in pipeline_param_keys:
        pipeline_kwargs['guidance_scale'] = kwargs['guidance_scale']
    if 'ddim' in solver and hasattr(pipeline, 'vqvae'):
        assert 'skip_decode' in pipeline_param_keys, ValueError(f"{type(pipeline)} not support output latent image generation.")
        pipeline_kwargs['skip_decode'] = True

    condition_sample = None
    if 'class_labels' in pipeline_param_keys:
        num_class = pipeline.embedder.config.num_class
        match kwargs['condition']:
            case 'uniform':
                condition_sample = torch.arange(0, sample_size, dtype=torch.long) * num_class // sample_size
                condition_sample = condition_sample[torch.randperm(sample_size, generator=generator)]
                condition_sample = condition_sample.to(pipeline.device)
            case 'random':
                condition_sample = None

    for index, size in split_batch(sample_size, batch_size):
        if condition_sample is not None:
            output, = pipeline(batch_size=size, class_labels=condition_sample[index:index + size], **pipeline_kwargs)
        else:
            output, = pipeline(batch_size=size, **pipeline_kwargs)
        yield index_wrapper(index, index + size), output

def main():
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.autograd.set_detect_anomaly(True)

    opts = parse_options()

    LOGGER.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    LOGGER.addHandler(console_handler)

    device = opts.device

    model_path = opts.model

    LOGGER.info(f"Loading pipeline with config {opts.__dict__}")

    match opts.solver:
        case 'any':
            pipeline = DiffusionPipeline.from_pretrained(model_path)
        case 'ddim':
            pipeline = DDIMPipeline.from_pretrained(model_path)
        case 'ddim-c':
            pipeline = DiffusionPipeline.from_pretrained(model_path)
        case _:
            raise ValueError
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    
    sample_size = opts.sample_size
    batch_size = opts.batch_size
    format = opts.format

    out_path = opts.output
    if out_path.find('?') != -1:
        subdir = time.strftime("%Y%m%d-%H%M%S")
        out_path = out_path.replace('?', subdir)
        subdir_index = 2
        while os.path.exists(out_path):
            out_path = out_path.replace('?', subdir + '-' + subdir_index)
            subdir_index += 1
        del subdir_index
    os.makedirs(out_path)

    if opts.dump_config:
        LOGGER.info("Generation config: %s", opts)

        with open(os.path.join(out_path, 'config.json'), 'w') as f:
            json.dump(opts.__dict__, f)

    os.makedirs(out_path, exist_ok=True)

    with tqdm(sample(pipeline, **opts.__dict__), "Generating samples", total=(sample_size + batch_size - 1) // batch_size, unit='batch') as sampler:
        if format in ('npy', 'npz'):
            itr = iter(sampler)
            try:
                indices, images = next(itr)
                buffer = np.empty((sample_size, *images.shape[1:]), dtype=np.uint8)
                while True:
                    buffer[indices] = np.round(images * 255).astype(np.uint8)
                    indices, images = next(itr)
            except StopIteration:
                pass
            if format == 'npz':
                np.savez(os.path.join(out_path, 'images.npz'), buffer)
            else:
                np.save(os.path.join(out_path, 'images.npy'), buffer)
        else:
            if opts.dump_config:
                image_path = os.path.join(out_path, 'images')
                os.mkdir(image_path)
            else:
                image_path = out_path
            for indices, images in sampler:
                for index, image in zip(indices, images):
                    image.save(os.path.join(image_path, f"{index}.{format}"))

    LOGGER.info(f"Generate samples at: {out_path}")

if __name__ == '__main__':
    main()