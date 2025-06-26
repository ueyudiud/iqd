import torch
from torch.utils.data import Dataset, Sampler


class BalancedRandomTimeEmbedDataset(Dataset):
    def __init__(self, dataset, num_steps):
        self.dataset = dataset
        self.size = len(dataset)
        self._num_steps_sub_one = num_steps - 1

    def __getitem__(self, index):
        if index < 0:
            index += self.size * 2
            if index < 0:
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )

        return self.dataset[index] if index < self.size else self._num_steps_sub_one - self.dataset[index - self.size]

    def __len__(self):
        return self.size * 2


class BalancedTimeEmbedRandomSampler(Sampler):
    def __init__(self, dataset: BalancedRandomTimeEmbedDataset):
        super().__init__()
        self.size = dataset.size

    def __iter__(self):
        indices = torch.randperm(self.size).tolist()
        for index in indices:
            yield index
            yield index + self.size

    def __len__(self):
        return self.size * 2
