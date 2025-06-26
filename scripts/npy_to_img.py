import os
from argparse import ArgumentParser
from PIL import Image

import numpy as np
from tqdm import tqdm

AVAILABLE_EXTENSIONS = ['jpg', 'jpeg', 'png']

def parse_options():
    parser = ArgumentParser()

    parser.add_argument('source', type=str)

    parser.add_argument('--output', '-o', type=str)

    parser.add_argument('--size', '-s', type=int, default=None)

    parser.add_argument('--format', choices=AVAILABLE_EXTENSIONS, default='jpg')

    return parser.parse_args()

opts = parse_options()

source = opts.source

array = np.load(source).files['arr_0']

output = opts.output
format = opts.format

os.makedirs(output, exist_ok=True)

print(f"Output array shape is {array.shape}")
for index, image in enumerate(tqdm(array, unit='images')):
    Image.fromarray(image).save(f"{output}/{index}.{format}")

print(f"Save images at {output}")