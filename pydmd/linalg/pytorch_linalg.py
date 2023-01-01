import logging

logging.basicConfig(format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
logger = logging.getLogger(__name__)

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
    def atleast_1d(cls, X):
        import torch

        return torch.atleast_1d(X)

    @classmethod
    def atleast_2d(cls, X):
        import torch

        return torch.atleast_2d(X)

    @classmethod
    def argsort(cls, X, *args, **kwargs):
        import torch

        if not torch.is_complex(X):
            return torch.argsort(X, *args, **kwargs)
        raise NotImplementedError("This feature is not supported in PyTorch")

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

        if torch.is_complex(X) and not torch.is_complex(Y):
            logger.info(f"Y dtype is not complex, casting to {X.dtype}")
            Y = Y.type(X.dtype)
        elif torch.is_complex(Y) and not torch.is_complex(X):
            logger.info(f"X dtype is not complex, casting to {Y.dtype}")
            X = X.type(Y.dtype)
        return torch.matmul(X, Y)

    @classmethod
    def eig(cls, X):
        import torch

        return torch.linalg.eig(X)

    @classmethod
    def full(cls, size, fill_value, *args, **kwargs):
        import torch

        if isinstance(size, int):
            size = (size,)
        return torch.full(size, fill_value, *args, **kwargs)

    @classmethod
    def hstack(cls, Xs):
        import torch

        return torch.hstack(Xs)

    @classmethod
    def inv(cls, X):
        import torch

        return torch.linalg.inv(X)

    @classmethod
    def isnan(cls, X):
        import torch

        return torch.isnan(X)

    @classmethod
    def log(cls, X):
        import torch

        return torch.log(X)

    @classmethod
    def lstsq(cls, X, Y, rcond):
        import torch

        if Y.ndim == 1:
            solution = torch.linalg.lstsq(X, Y[:,None], rcond=rcond).solution
            return torch.squeeze(solution)
        return torch.linalg.lstsq(X, Y, rcond=rcond).solution

    @classmethod
    def make_not_writeable(cls, X):
        # not supported
        logger.info("PyTorch does not support non-writeable tensors, ignoring ...")

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

        complex_dtypes = [arr.dtype for arr in Xs if torch.is_complex(arr)]
        if complex_dtypes:
            complex_dtypes.sort(key=lambda dtype: torch.finfo(dtype).bits)
            logger.info(f"Converting tensors to {complex_dtypes[-1]}")
            Xs = tuple(map(lambda X: X.type(complex_dtypes[-1]), Xs))
        return torch.linalg.multi_dot(Xs, *args, **kwargs)

    @classmethod
    def nanmean(cls, X, axis):
        import torch
        
        if torch.is_complex(X):
            real = torch.nanmean(X.real, dim=axis)
            imag = torch.nanmean(X.imag, dim=axis)
            return torch.complex(real, imag)
        return torch.nanmean(X, dim=axis)

    @classmethod
    def nansum(cls, X, axis):
        import torch
        
        if torch.is_complex(X):
            real = torch.nansum(X.real, dim=axis)
            imag = torch.nansum(X.imag, dim=axis)
            return torch.complex(real, imag)
        return torch.nansum(X, dim=axis)

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
    def split(cls, X, n_arrays, axis):
        import torch

        if not isinstance(n_arrays, int):
            raise ValueError(
                "The only supported split strategy at the moment is splitting in arrays of same size"
            )
        return torch.tensor_split(X, n_arrays, axis)

    @classmethod
    def sqrtm(cls, X):
        import torch

        return torch.pow(X, 0.5)

    @classmethod
    def svd(cls, X, *args, **kwargs):
        import torch

        return torch.linalg.svd(X, *args, **kwargs)

    @classmethod
    def to(cls, reference, *args):
        import torch

        target_device = reference.device
        transformed = []

        for X in args:
            module = X.__class__.__module__
            if module == "numpy":
                X_transformed = torch.from_numpy(X).to(
                    target_device, dtype=reference.dtype
                )
            elif module == "torch":
                X_transformed = X.to(target_device, dtype=reference.dtype)
            elif module.startswith("scipy.sparse"):
                if "coo" in module:
                    i = torch.LongTensor(np.stack((X.row, X.col)))
                    v = torch.FloatTensor(X.data)
                    X_transformed = torch.sparse_coo_tensor(
                        i,
                        v,
                        dtype=reference.dtype,
                        size=X.shape,
                        device=target_device,
                    )
                elif "csr" in module:
                    X_transformed = torch.sparse_csr_tensor(
                        X.indptr,
                        X.indices,
                        X.data,
                        dtype=reference.dtype,
                        size=X.shape,
                        device=target_device,
                    )
                else:
                    raise ValueError(f"Unsupported sparse matrix type: {type(X)}")
            else:
                raise ValueError(f"Unsupported module type: {module}")
            transformed.append(X_transformed)

        if len(transformed) == 1:
            return transformed[0]
        return transformed

    @classmethod
    def vander(cls, X, N, increasing):
        import torch

        return torch.vander(X, N, increasing)

    @classmethod
    def vstack(cls, Xs):
        import torch

        return torch.vstack(Xs)
