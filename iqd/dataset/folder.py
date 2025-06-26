from os import walk
from os.path import expanduser, join

import torchvision
from PIL import Image


def default_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def _make_dataset(dir):
    dir = expanduser(dir)
    instances = []
    for root, _, fnames in walk(dir):
        for fname in sorted(fnames):
            path = join(root, fname)
            instances.append(path)
    return instances


class ImageFolder(torchvision.datasets.VisionDataset):
    def __init__(self, root, loader=default_loader, transforms=None, clip=True, _sub=None):
        super().__init__(root, transforms)
        self.loader = loader
        self.samples = _make_dataset(root) if _sub is None else _sub
        self.clip = clip

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)
        if self.transforms is not None:
            sample = self.transforms(sample)
        if self.clip:
            sample = sample * 2 - 1
        return sample

    def __len__(self):
        return len(self.samples)

    def slice(self, index):
        return ImageFolder(self.root, self.loader, self.transforms, self.clip, self.samples[index])
