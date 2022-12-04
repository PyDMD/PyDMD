import functools
import logging

from .linalg_base import LinalgBase

# we can assume that NumPy is installed
import numpy as np


@functools.lru_cache(maxsize=None)
class LinalgNumPy(LinalgBase):
    @classmethod
    def abs(cls, X):
        return np.abs(X)

    @classmethod
    def append(cls, X, Y, axis):
        return np.append(X, Y, axis)

    @classmethod
    def arange(cls, *args, **kwargs):
        return np.arange(*args, **kwargs)

    @classmethod
    def atleast_2d(cls, X):
        return np.atleast_2d(X)

    @classmethod
    def ceil(cls, X):
        return np.ceil(X)

    @classmethod
    def cond(cls, X):
        return np.linalg.cond(X)

    @classmethod
    def diag(cls, X):
        return np.diag(X)

    @classmethod
    def eig(cls, X):
        return np.linalg.eig(X)

    @classmethod
    def full(cls, size, fill_value):
        return np.full(size, fill_value)

    @classmethod
    def inv(cls, X):
        return np.linalg.inv(X)

    @classmethod
    def log(cls, X):
        return np.log(X)

    @classmethod
    def lstsq(cls, X, Y, rcond):
        return np.linalg.lstsq(X, Y, rcond=rcond)

    @classmethod
    def make_not_writeable(cls, X):
        X.flags.writeable = False

    @classmethod
    def median(cls, X, *args, **kwargs):
        return np.median(X)

    @classmethod
    def multi_dot(cls, Xs, *args, **kwargs):
        return np.linalg.multi_dot(Xs, *args, **kwargs)

    @classmethod
    def new_array(cls, X):
        return np.array(X)

    @classmethod
    def norm(cls, Xs, *args, **kwargs):
        return np.linalg.norm(X, *args, **kwargs)

    @classmethod
    def pow(cls, X, power):
        return np.power(X, power)

    @classmethod
    def repeat(cls, X, repeats, axis):
        return np.repeat(X, repeats, axis=axis)

    @classmethod
    def searchsorted(cls, X, val, *args, **kwargs):
        return np.searchsorted(X, val)

    @classmethod
    def solve(cls, A, b):
        return np.linalg.solve(A, b)

    @classmethod
    def sqrtm(cls, X):
        from scipy.linalg import sqrtm

        return sqrtm(X)

    @classmethod
    def svd(cls, X, *args, **kwargs):
        return np.linalg.svd(X, *args, **kwargs)

    @classmethod
    def to(cls, X, other_numpy_array):
        # in general it's not critical to convert X to NumPy because the
        # operation is already handled quietly by all the frameworks
        # TODO implement properly if this generates problems
        logging.info("to(X, other_numpy_array) ignored quietly")
        return X

    @classmethod
    def vander(cls, X, N, increasing):
        return np.vander(X, N, increasing)
