import argparse
import io
import json
import logging
import os
import sys
from functools import partial
from os.path import join as join_path

import numpy as np
import torch
import torchvision.transforms
from diffusers import DiffusionPipeline, DDIMPipeline

from iqd.dataset import ImageFolder
from iqd.dataset.np_to_pt import NumpyImageDataset
from iqd.dataset.tensor import TensorDataset
from iqd.quantize.quantizer import IQDQuantizer, LSQQuantizer, LinearQuantizer

LOGGER = logging.getLogger(__name__)

def make_validate_callback(log_dir, seed=None):
    seed = torch.seed() if seed is None else seed
    generator = torch.Generator()
    path = join_path(log_dir, 'validate')
    os.makedirs(path, exist_ok=True)

    @torch.no_grad()
    def validate(pipeline: DDIMPipeline, num_stage, num_step, **kwargs):
        generator.manual_seed(seed)
        samples = pipeline(16, num_inference_steps=100, generator=generator, return_type='np').images
        np.savez(f"{path}/{num_step}.npz", samples)
        
        # # from PIL import Image
        # # import numpy as np
        #
        # with torch.no_grad():
        #     generator.manual_seed(seed)
        #     # images = tuple(pipeline(batch_size=3, generator=generator, return_dict=False)[0] for _ in range(3))
        #     # image = np.concatenate(np.concatenate(np.array(images), axis=1), axis=1)
        #     # image = Image.fromarray(image)
        #     image, = pipeline(num_inference_steps=100, generator=generator, return_dict=False)[0]
        #
        # image_path = os.path.join(path, f"{num_stage}-{num_step}.jpg")
        # image.save(image_path)
    
    return validate

def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--device', 
        default='cuda',
        help="specify training device"
    )

    parser.add_argument(
        '--method',
        default='iqd',
        choices=['iqd', 'lsq', 'linear']
    )

    parser.add_argument(
        'source',
        help="choose origin model"
    )

    parser.add_argument(
        '--dataset',
        default='!empty'
    )

    parser.add_argument(
        '--calib-dataset',
        default='!generated'
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

    parser.add_argument(
        '--cali-sample-size',
        type=int,
        default=16
    )

    parser.add_argument(
        '--solver',
        choices=['any', 'ddim', 'ddpm'],
        default='any'
    )

    parser.add_argument(
        '--legacy',
        action='store_true',
    )

    sampler_group = parser.add_argument_group(
        'sampler options'
    )

    sampler_group.add_argument(
        '--sample-size',
        type=int,
        default=10000
    )

    quant_group = parser.add_argument_group(
        'quantization options'
    )

    quant_group.add_argument(
        '--quant-weight-bits',
        type=int,
        default=8)

    quant_group.add_argument(
        '--quant-activation-bits',
        type=int,
        default=None)

    quant_group.add_argument(
        '--lr',
        type=float,
        default=1e-5)

    quant_group.add_argument(
        '--lr-decay',
        type=float,
        default=0.75)

    quant_group.add_argument(
        '--train-config',
        default=None)

    quant_group.add_argument(
        '--decay',
        type=float,
        default=0.0)

    quant_group.add_argument(
        '--weight-schedule',
        default='1',
        choices=['1', 'snr', 'max-snr-1', 'min-snr-5', 'diff-snr'])

    quant_group.add_argument(
        '--loss-schedule',
        default='l2',
        choices=['l1', 'l2', 'lpips', 'l2fft'])

    return parser.parse_args()

def load_dataset(path: str, size: int | None, opts, pipeline: DDIMPipeline):
    match path:
        case '!empty':
            image_shape = (
                pipeline.unet.config.in_channels,
                pipeline.unet.config.sample_size,
                pipeline.unet.config.sample_size,
            )
            return TensorDataset(torch.zeros((size, *image_shape)))
        case '!generated':
            assert size is not None, ValueError("batch size is required for generated dataset.")

            batch_size = opts.batch_size
            generator = torch.Generator().manual_seed(2)

            generate = partial(pipeline,
                               generator=generator,
                               num_inference_steps=pipeline.scheduler.config.num_train_timesteps,
                               output_type='pt')

            samples = []
            count = 0

            LOGGER.info("Generating samples for dataset.")
            while count + batch_size < size:
                batched_samples, = generate(batch_size)
                samples.append(batched_samples)
                count += batch_size
            batched_samples, = generate(size - count)
            samples.append(batched_samples)
            return TensorDataset(torch.stack(samples))
        case _:
            if not os.path.isdir(path):
                dataset = np.load(path)
                if not isinstance(dataset, np.ndarray):
                    dataset = dataset['arr_0']
                if size is not None:
                    dataset = dataset[:size]
                return NumpyImageDataset(dataset)
            else:
                dataset = ImageFolder(path, transforms=torchvision.transforms.ToTensor())
                if size is not None:
                    dataset = dataset.slice(slice(None, size))
                return dataset

def main():
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.utils.deterministic.fill_uninitialized_memory = True

    opts = parse_options()

    LOGGER.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    LOGGER.addHandler(console_handler)

    device = opts.device

    torch.manual_seed(29475)

    quantizer_path = opts.train_config
    if quantizer_path is not None:
        LOGGER.info(f"Loading train config from {quantizer_path}")
        with io.open(quantizer_path) as fp:
            quantizer_kwargs = json.load(fp)
    else:
        quantizer_kwargs = None

    match opts.method:
        case 'iqd':
            Quantizer = IQDQuantizer
            if quantizer_kwargs is None:
                quantizer_kwargs = {
                    'num_timestep': 100,
                    'lambda_r': 0,
                    'init_amode': 'ema-saturate',
                    'lr': opts.lr,
                    'lr_decay': opts.lr_decay,
                    'weight_schedule': opts.weight_schedule,
                    'stages': [
                        # {'apr': 1.0, 'wpr': 1. - 0.5 ** 0, 'epoch': 5},
                        # {'apr': 1.0, 'wpr': 1. - 0.5 ** 1, 'epoch': 5},
                        # {'apr': 1.0, 'wpr': 1. - 0.5 ** 2, 'epoch': 5},
                        # {'apr': 1.0, 'wpr': 1. - 0.5 ** 3, 'epoch': 5},
                        {'apr': 1.0, 'wpr': 1., 'epoch': 1},
                    ]
                }
        case 'lsq':
            Quantizer = LSQQuantizer
            if quantizer_kwargs is None:
                quantizer_kwargs = {
                    'num_timestep': 100,
                    'lambda_r': 0,
                    'init_amode': 'ema-minmax',
                    'lr': opts.lr,
                    'lr_decay': opts.lr_decay
                }
        case 'linear':
            Quantizer = LinearQuantizer
            if quantizer_kwargs is None:
                quantizer_kwargs = {
                    'num_timestep': 100,
                    'lambda_r': 0,
                    'init_amode': 'ema-saturate',
                }
        case _:
            raise ValueError

    quant_suffix = f"w{opts.quant_weight_bits}a{opts.quant_activation_bits}"
    quantizer = Quantizer.from_config(
        quantizer_kwargs,
        wbits=opts.quant_weight_bits,
        abits=opts.quant_activation_bits,
    )

    source_path = opts.source

    LOGGER.info(f"Loading pipeline from {source_path}")

    pipeline = DiffusionPipeline.from_pretrained(source_path)

    out_path = join_path(opts.out_dir, opts.out_name or os.path.basename(source_path) + "-" + quant_suffix)

    os.makedirs(out_path, exist_ok=True)

    config_path = join_path(out_path, "config.json")
    LOGGER.info('Saving train configuration to %s', config_path)
    with io.open(config_path, 'w') as fp:
        json.dump(opts.__dict__, fp)

    callback = make_validate_callback(out_path, seed=12306)

    train_dataset = load_dataset(opts.dataset, opts.sample_size, opts, pipeline)
    calib_dataset = load_dataset(opts.calib_dataset, opts.cali_sample_size, opts, pipeline)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    pipeline, profile = quantizer.quantize(
        pipeline,
        train_samples=train_dataset,
        calib_samples=calib_dataset,
        batch_size=opts.batch_size,
        callback=callback,
        generator=torch.Generator().manual_seed(4568),
        device=device,
        logger=LOGGER,
    )
    end_event.record()
    LOGGER.info(f"Takes {start_event.elapsed_time(end_event)} times.")

    model_path = join_path(out_path, 'model')
    LOGGER.info('Saving model to %s', model_path)
    pipeline.save_pretrained(model_path)

    profile_path = join_path(out_path, 'profile.json')

    LOGGER.info('Saving training profile to %s', profile_path)
    with io.open(profile_path, 'w') as fp:
        json.dump(profile, fp)

if __name__ == '__main__':
    main()
