import os
from argparse import ArgumentParser
from PIL import Image

import numpy as np
from tqdm import tqdm

AVAILABLE_EXTENSIONS = ['.jpg', '.jpeg', '.png']

def parse_options():
    parser = ArgumentParser()

    parser.add_argument('source', type=str)

    parser.add_argument('--output', '-o', type=str)

    parser.add_argument('--size', '-s', type=int, default=None)

    parser.add_argument('--recursive', '-r', type=bool, default=True)

    return parser.parse_args()

opts = parse_options()

source = opts.source

if opts.recursive:
    def list_files():
        for name, _, files in os.walk(source):
            for file in files:
                yield os.path.join(name, file)
    files = list_files()
else:
    files = os.listdir(source)

files = [file for file in files if any(file.endswith(ext) for ext in AVAILABLE_EXTENSIONS)]

size = opts.size
if size is not None:
    files = files[:size]

image = np.asarray(Image.open(files[0]))
shape = (len(files), *image.shape)
print(f"Output array shape is {shape}")
array = np.empty(shape, dtype=np.uint8)

with tqdm(total=len(files), unit='images') as progress:
    array[0] = image
    progress.update()

    for index, file in enumerate(files[1:], 1):
        image = np.asarray(Image.open(files[0]))
        if shape[1:] != image.shape:
            raise ValueError('Size of all images must be same.')
        array[index] = image
        progress.update()

output = opts.output
if output is None:
    output = source + '.npz'

np.savez(output, array)

print(f"Save .npz file at {output}")