import logging
from scipy.linalg import sqrtm, block_diag

logging.basicConfig(
    format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
)

from .linalg_base import LinalgBase

# we can assume that NumPy is installed
import numpy as np


class LinalgNumPy(LinalgBase):
    def __init__(self):
        raise RuntimeError("Instances not allowed")

    @classmethod
    def abs(cls, X):
        return np.abs(X)

    @classmethod
    def append(cls, X, Y, axis):
        return np.append(X, Y, axis)

    @classmethod
    def arange(cls, *args, **kwargs):
        if "device" in kwargs:
            if kwargs["device"] != "cpu":
                raise ValueError("Cannot create NumPy array on GPU")
            del kwargs["device"]
        return np.arange(*args, **kwargs)

    @classmethod
    def atleast_1d(cls, X):
        return np.atleast_1d(X)

    @classmethod
    def atleast_2d(cls, X):
        return np.atleast_2d(X)

    @classmethod
    def argsort(cls, X, *args, **kwargs):
        return np.argsort(X, *args, **kwargs)

    @classmethod
    def array_split(cls, input, indices_or_sections, axis=0):
        return np.array_split(
            input, indices_or_sections=indices_or_sections, axis=axis
        )

    @classmethod
    def block_diag(cls, *arrs):
        return block_diag(*arrs)

    @classmethod
    def cat(cls, Xs, axis):
        return np.concatenate(Xs, axis=axis)

    @classmethod
    def ceil(cls, X):
        return np.ceil(X)

    @classmethod
    def cond(cls, X):
        return np.linalg.cond(X)

    @classmethod
    def device(cls, X):
        return "cpu"

    @classmethod
    def diag_matrix(cls, X):
        return np.diag(X)

    @classmethod
    def dot(cls, X, Y):
        if X.ndim == 2 and Y.ndim == 3:
            return np.matmul(X, Y)
        return X.dot(Y)

    @classmethod
    def eig(cls, X):
        return np.linalg.eig(X)

    @classmethod
    def extract_diagonal(cls, X):
        return np.diag(X)

    @classmethod
    def full(cls, size, fill_value, *args, **kwargs):
        return np.full(size, fill_value, *args, **kwargs)

    @classmethod
    def inv(cls, X):
        return np.linalg.inv(X)

    @classmethod
    def isnan(cls, X):
        return np.isnan(X)

    @classmethod
    def log(cls, X):
        return np.log(X)

    @classmethod
    def lstsq(cls, X, Y, rcond):
        return np.linalg.lstsq(X, Y, rcond=rcond)[0]

    @classmethod
    def make_not_writeable(cls, X):
        X.flags.writeable = False

    @classmethod
    def median(cls, X, *args, **kwargs):
        return np.median(X)

    @classmethod
    def moveaxis(cls, X, source, destination):
        return np.moveaxis(X, source, destination)

    @classmethod
    def multiply_elementwise(cls, X, Y):
        return np.multiply(X, Y)

    @classmethod
    def multi_dot(cls, Xs, *args, **kwargs):
        return np.linalg.multi_dot(Xs, *args, **kwargs)

    @classmethod
    def nanmean(cls, X, axis):
        return np.nanmean(X, axis=axis)

    @classmethod
    def nansum(cls, X, axis):
        return np.nansum(X, axis=axis)

    @classmethod
    def new_array(cls, X):
        return np.array(X)

    @classmethod
    def matrix_norm(cls, X, *args, **kwargs):
        return np.linalg.norm(X, *args, **kwargs)

    @classmethod
    def pinv(cls, X):
        import scipy

        return scipy.linalg.pinv(X)

    @classmethod
    def pow(cls, X, power):
        return np.power(X, power)

    @classmethod
    def pseudo_hankel_matrix(cls, X, d):
        return np.vstack(
            tuple(X[:, i : i + (X.shape[1] + 1 - d)] for i in range(d))
        )

    @classmethod
    def qr_reduced(cls, X):
        return np.linalg.qr(X, mode="reduced")

    @classmethod
    def random(cls, shape):
        return np.random.rand(*shape)

    @classmethod
    def repeat(cls, X, repeats, axis):
        return np.repeat(X, repeats, axis=axis)

    @classmethod
    def reshape(cls, X, shape):
        return X.reshape(shape)

    @classmethod
    def searchsorted(cls, X, val, *args, **kwargs):
        return np.searchsorted(X, val)

    @classmethod
    def solve(cls, A, b):
        return np.linalg.solve(A, b)

    @classmethod
    def split(cls, X, n_arrays, axis):
        if not isinstance(n_arrays, int):
            raise ValueError(
                "The only supported split strategy at the moment is splitting in arrays of same size"
            )
        return np.split(X, n_arrays, axis)

    @classmethod
    def matrix_sqrt(cls, X):
        X2 = sqrtm(X)
        if hasattr(np, "complex256") and X2.dtype == np.complex256:
            X2 = X2.astype(np.complex128)
            msg = "Casting atilde from np.complex256 to np.complex128"
            logging.info(msg)
        return X2

    @classmethod
    def svd(cls, X, *args, **kwargs):
        return np.linalg.svd(X, *args, **kwargs)

    @classmethod
    def to(cls, reference, *args):
        args = tuple(arg.astype(reference.dtype) for arg in args)
        
        if len(args) == 1:
            return args[0]
        return args

    @classmethod
    def vander(cls, X, N, increasing):
        return np.vander(X, N, increasing)
