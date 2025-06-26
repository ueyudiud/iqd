import torch
from torch import Generator
from torch.utils.data import Dataset


class _GeneratedDataset(Dataset):
    def __init__(self, sample_size, shape, group_size, unfix=False):
        super().__init__()
        self.shape = shape
        self.sample_size = sample_size
        self.group_count = (sample_size + group_size - 1) // group_size
        self.group_size = group_size
        self.unfix = unfix

    def _post_init(self, seed):
        generator = Generator().manual_seed(seed if seed is not None else torch.seed())
        self.generator = generator

        if self.unfix:
            return

        states = []
        out = torch.empty(self.shape)

        for _ in range(self.group_count - 1):
            states.append(generator.get_state())
            for _ in range(self.group_size):
                self._generate(generator, out)
        states.append(generator.get_state())

        self.states = torch.stack(states)

    def _generate(self, generator, out):
        raise NotImplemented

    def __getitem__(self, index):
        if index < 0:
            index += self.sample_size
        if index >= self.sample_size or index < 0:
            raise ValueError("data index out of bound")

        out = torch.empty(self.shape)

        if self.unfix:
            return self._generate(self.generator, out)

        group_index = index // self.group_size
        sub_index = index % self.group_size
        generator = self.generator.set_state(self.states[group_index])
        for _ in range(sub_index):
            self._generate(generator, out)
        return self._generate(generator, out)

    def __len__(self):
        return self.sample_size


class GeneratedUniformIntegerDataset(_GeneratedDataset):
    high: int

    def __init__(self, high, sample_size, shape, group_size=64, unfix=False, seed=None):
        super().__init__(sample_size, shape, group_size, unfix)
        self.high = high
        self._post_init(seed)

    def _generate(self, generator, out):
        return torch.randint(self.high, self.shape, generator=generator, out=out)


class GeneratedUniformDataset(_GeneratedDataset):
    def __init__(self, sample_size, shape, group_size=64, unfix=False, seed=None):
        super().__init__(sample_size, shape, group_size, unfix)
        self._post_init(seed)

    def _generate(self, generator, out):
        return torch.rand(self.shape, generator=generator, out=out)


class GeneratedGaussianDataset(_GeneratedDataset):
    def __init__(self, sample_size, shape, group_size=64, unfix=False, seed=None):
        super().__init__(sample_size, shape, group_size, unfix)
        self._post_init(seed)

    def _generate(self, generator, out):
        return torch.randn(self.shape, generator=generator, out=out)
