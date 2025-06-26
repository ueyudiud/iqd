import importlib


def fetch_class_library_tuple(value):
    library = value.__module__.split(".")[0]
    class_name = value.__class__.__name__
    return library, class_name


def load_class(library, class_name):
    module = importlib.import_module(library)
    return getattr(module, class_name)