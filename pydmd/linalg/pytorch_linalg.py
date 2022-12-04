import functools
import logging

from .linalg_base import LinalgBase

import numpy as np


@functools.lru_cache(maxsize=None)
class LinalgPyTorch(LinalgBase):
    @classmethod
    def abs(cls, X):
        return X.abs()

    @classmethod
    def append(cls, X, Y, axis):
        import torch

        return torch.cat((X, Y), axis)

    @classmethod
    def arange(cls, *args, **kwargs):
        import torch

        return torch.arange(*args, **kwargs)

    @classmethod
    def atleast_2d(cls, X):
        import torch

        return torch.atleast_2d(X)

    @classmethod
    def ceil(cls, X):
        import torch

        return torch.ceil(X)

    @classmethod
    def cond(cls, X):
        import torch

        return torch.linalg.cond(X)

    @classmethod
    def diag(cls, X):
        import torch

        return torch.diag(X)

    @classmethod
    def eig(cls, X):
        import torch

        return torch.linalg.eig(X)

    @classmethod
    def full(cls, size, fill_value):
        import torch

        if isinstance(size, int):
            size = (size,)
        return torch.full(size, fill_value)

    @classmethod
    def inv(cls, X):
        import torch

        return torch.linalg.inv(X)

    @classmethod
    def log(cls, X):
        import torch

        return torch.log(X)

    @classmethod
    def lstsq(cls, X, Y, rcond):
        import torch

        return torch.linalg.lstsq(X, Y, rcond=rcond)

    @classmethod
    def make_not_writeable(cls, X):
        # not supported
        logging.info(
            "PyTorch does not support non-writeable tensors, ignoring ..."
        )

    @classmethod
    def median(cls, X, *args, **kwargs):
        import torch

        return torch.median(X)

    @classmethod
    def multi_dot(cls, Xs, *args, **kwargs):
        import torch

        return torch.linalg.multi_dot(Xs, *args, **kwargs)

    @classmethod
    def new_array(cls, X):
        import torch

        return torch.tensor(X)

    @classmethod
    def norm(cls, Xs, *args, **kwargs):
        import torch

        return torch.linalg.norm(X, *args, **kwargs)

    @classmethod
    def pow(cls, X, power):
        import torch

        return torch.pow(X, power)

    @classmethod
    def repeat(cls, X, repeats, axis):
        import torch

        return torch.repeat_interleave(X, repeats, dim=axis)

    @classmethod
    def searchsorted(cls, X, val, *args, **kwargs):
        import torch

        return torch.searchsorted(X, val)

    @classmethod
    def solve(cls, A, b):
        import torch

        return torch.linalg.solve(A, b)

    @classmethod
    def sqrtm(cls, X):
        import torch

        return torch.pow(X, 0.5)

    @classmethod
    def svd(cls, X, *args, **kwargs):
        import torch

        return torch.linalg.svd(X, *args, **kwargs)

    @classmethod
    def to(cls, X, other_torch_array):
        import torch

        target_device = other_torch_array.device
        module = X.__class__.__module__
        if module == "numpy":
            return torch.from_numpy(X).to(
                target_device, dtype=other_torch_array.dtype
            )
        elif module == "torch":
            return X.to(target_device, dtype=other_torch_array.dtype)
        elif module == "scipy.sparse._coo":
            i = torch.LongTensor(np.stack((X.row, X.col)))
            v = torch.FloatTensor(X.data)
            return torch.sparse_coo_tensor(
                i,
                v,
                dtype=other_torch_array.dtype,
                size=X.shape,
                device=target_device,
            )
        elif module == "scipy.sparse._csr":
            return torch.sparse_csr_tensor(
                X.indptr,
                X.indices,
                X.data,
                dtype=other_torch_array.dtype,
                size=X.shape,
                device=target_device,
            )
        else:
            raise ValueError("Unsupported module type: {}".format(module))

    @classmethod
    def vander(cls, X, N, increasing):
        import torch

        return torch.vander(X, N, increasing)
