import logging

from .linalg_base import LinalgBase

import numpy as np


class LinalgPyTorch(LinalgBase):
    def __init__(self):
        raise RuntimeError("Instances not allowed")

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
    def dot(cls, X, Y):
        import torch

        return torch.matmul(X, Y)

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
    def hstack(cls, Xs):
        return torch.hstack(Xs)

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
    def multiply_elementwise(cls, X, Y):
        import torch

        return torch.mul(X, Y)

    @classmethod
    def multi_dot(cls, Xs, *args, **kwargs):
        import torch

        return torch.linalg.multi_dot(Xs, *args, **kwargs)

    @classmethod
    def new_array(cls, X):
        import torch

        if torch.is_tensor(X):
            return X
        if isinstance(X, (list, tuple)):
            if not X:
                return torch.zeros(0)
            if isinstance(X[0], (list, tuple)) or torch.is_tensor(X[0]):
                X = tuple(map(np.array, X))
        return torch.from_numpy(np.array(X))

    @classmethod
    def norm(cls, X, *args, **kwargs):
        import torch

        return torch.linalg.norm(X, *args, **kwargs)

    @classmethod
    def pinv(cls, X):
        import torch

        return torch.linalg.pinv(X)

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
        elif module.startswith("scipy.sparse"):
            if "coo" in module:
                i = torch.LongTensor(np.stack((X.row, X.col)))
                v = torch.FloatTensor(X.data)
                return torch.sparse_coo_tensor(
                    i,
                    v,
                    dtype=other_torch_array.dtype,
                    size=X.shape,
                    device=target_device,
                )
            elif "csr" in module:
                return torch.sparse_csr_tensor(
                    X.indptr,
                    X.indices,
                    X.data,
                    dtype=other_torch_array.dtype,
                    size=X.shape,
                    device=target_device,
                )
            raise ValueError("Unsupported sparse matrix type: {}", type(X))
        raise ValueError("Unsupported module type: {}".format(module))

    @classmethod
    def vander(cls, X, N, increasing):
        import torch

        return torch.vander(X, N, increasing)

    @classmethod
    def vstack(cls, Xs):
        import torch
        
        return torch.vstack(Xs)
