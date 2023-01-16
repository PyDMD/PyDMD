import functools

from .linalg_base import LinalgBase
from .numpy_linalg import LinalgNumPy
from .pytorch_linalg import LinalgPyTorch


__linalg_module_mapper = {
    "numpy": LinalgNumPy,
    "torch": LinalgPyTorch,
}


def _extract_module_name(X):
    return X.__class__.__module__


def build_linalg_module(obj):
    if not is_array(obj):
        raise ValueError(
            "Expected PyTorch Tensor or NumPy NDarray, received {}".format(
                type(obj)
            )
        )
    # don't use isinstance to avoid import statements which might not resolve
    return __linalg_module_mapper[_extract_module_name(obj)]


def assert_same_linalg_type(X, *args):
    module_names = map(_extract_module_name, args)
    unexpected_module_names = set(module_names) - {_extract_module_name(X)}
    if unexpected_module_names:
        types = unexpected_module_names + {_extract_module_name(X)}
        raise ValueError(f"Found types: {types}")


def is_array(X):
    module = _extract_module_name(X)
    if module == "torch":
        return X.__class__.__name__ == "Tensor"
    elif module == "numpy":
        return X.__class__.__name__ == "ndarray"
    return False


def cast_as_array(X):
    if is_array(X):
        return X
    if isinstance(X, (list, tuple)):
        if is_array(X[0]):
            linalg_module = build_linalg_module(X[0])
            return linalg_module.new_array(X)
        else:
            import numpy as np

            return np.array(X)
    else:
        raise ValueError(
            "The given value cannot be casted to a supported array type: {}".format(
                type(X)
            )
        )


def generic_linalg(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        obj = args[0]
        # this handles cases like multi_dot
        if isinstance(obj, (tuple, list)):
            obj = obj[0]
        return func(build_linalg_module(obj), *args, **kwargs)

    return wrapper


def no_torch(X):
    if _extract_module_name(X) == "torch":
        raise ValueError("PyTorch not supported with this DMD variant")
