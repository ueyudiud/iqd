from torch import Tensor
from torch.utils.data import Dataset


class TensorDataset(Dataset):
    def __init__(self, tensor: Tensor):
        self.tensor = tensor

    def __getitem__(self, index: int) -> Tensor:
        return self.tensor[index]

    def __len__(self):
        return self.tensor.size(0)