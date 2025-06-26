from torch.nn import Module


class ModuleQat(Module):
    def detach(self) -> Module:
        return self

