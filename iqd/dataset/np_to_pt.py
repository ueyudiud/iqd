import numpy as np
import torch
from torch.utils.data import Dataset


class NumpyImageDataset(Dataset):
    def __init__(self, data: np.ndarray):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.add(-1, torch.from_numpy(self.data[idx].transpose(2, 0, 1)).float(), alpha=2 / 255)
