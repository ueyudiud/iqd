from .base import Grouping
from .reduce import ReduceGroup


class FullGrouping(Grouping):
    def __init__(self):
        pass

    def group(self, shape):
        return FullGroup(shape)


class FullGroup(ReduceGroup):
    def __init__(self, shape = None):
        super().__init__(shape, None)
        self.reduce_dim = None
