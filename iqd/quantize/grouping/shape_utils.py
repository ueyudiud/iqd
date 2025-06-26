from collections.abc import Iterable

from typing_extensions import overload


class Shape(tuple):
    def __new__(cls, shape):
        if not isinstance(shape, Iterable):
            shape = (shape,)

        self = super().__new__(cls, shape)

        n = len(self)
        n_e = None
        for i, l in enumerate(self):
            if l is ...:
                if n_e is not None:
                    raise ValueError("Contains two ellipsis")
                n_e = i

        self._upper = n_e or n
        self._lower = n_e + 1 - n if n_e is not None else 1 - n

        return self

    def sizes(self):
        n = len(self)
        n_e = self._upper

        for i in range(n_e):
            yield i, self[i]

        for i in range(n_e + 1, n):
            yield i - n, self[i]

    @overload
    def size(self, /) -> int: ...

    @overload
    def size(self, index: int, /) -> int: ...

    @overload
    def size(self, start: int, end: int, /) -> int: ...

    def size(self, start = None, end = None, /) -> int:
        if start is None:
            start = 0
            end = -1

        if end is None:
            if 0 <= start < self._upper:
                return self[start]

            if 0 > start >= self._lower:
                return self[start]
        else:
             if start >= 0 and end < 0:
                 if start < self._upper and end >= self._lower:
                     return -1
             else:
                 if start <= end:
                     s = 1
                     for l in self[start:end]:
                         s *= l
                     return s

        raise IndexError("Index out of shape range.")

    def reduce(self):
        return Shape(1 if isinstance(l, int) and l < 0 else l for l in self)

    def replace(self, dim: int, len: int):
        return Shape((*self[:dim], len, *self[dim + 1:]))

    def flat(self, start: int, end: int) -> "Shape":
        if not (
                (0 <= start <= self._upper and self._lower <= end < 0) or
                (start <= end and (start >= 0 or end < 0))
        ):
            raise IndexError("Index out of shape range.")

        return Shape((*self[:start], self.size(start, end), *self[end:]))

    def unflat(self, index: int, sizes: Iterable[int]) -> "Shape":
        if not (self._lower <= index < self._upper):
            raise IndexError("Index out of shape range.")
        # TODO
        return Shape((*self[:index], *sizes, *self[int:]))


def into_shape(shape: int | tuple[int, ...]) -> Shape:
    return Shape(shape)


