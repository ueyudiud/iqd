from types import FunctionType, MethodType, CellType
from typing import Any


def wrap(
        prototype: FunctionType,
        globals: dict[str, Any] | None = None,
        name: str | None = None,
        binding: tuple[CellType] | dict[str, Any] | None = None) -> FunctionType:
    globals = globals if globals is not None else dict(prototype.__globals__)
    if name is None:
        name = prototype.__name__
    if binding is None:
        closure = prototype.__closure__
    elif not isinstance(binding, tuple):
        closure = tuple(
            (CellType(binding[var]) if var in binding else cell)
            for var, cell in zip(prototype.__code__.co_freevars, prototype.__closure__)
        )
    else:
        closure = binding
    return FunctionType(code=prototype.__code__, globals=globals, name=name, argdefs=prototype.__defaults__,
                        closure=closure)


def wrap_init(T: type, globals: dict[str, Any] = None) -> MethodType:
    return wrap(T.__init__, globals=globals, name='__init__', binding={'__class__': T})
