import functools
import logging

from .linalg_base import LinalgBase


@functools.lru_cache(maxsize=None)
class LinalgSciPySparse(LinalgBase):
    @classmethod
    def abs(cls, X):
        import numpy as np

        return np.abs(X)

    @classmethod
    def append(cls, X, Y, axis):
        import scipy.sparse as sp

        if sp.issparse(X) and sp.issparse(Y):
            if axis == 0:
                return sp.vstack((X, Y))
            elif axis == 1:
                return sp.hstack((X, Y))
            raise ValueError(
                "Appending sparse matrices across axis {} is not supported".format(
                    axis
                )
            )
        raise ValueError("LinalgSciPySparse requires sparse matrices")

    @classmethod
    def arange(cls, *args, **kwargs):
        import numpy as np

        return np.arange(*args, **kwargs)

    @classmethod
    def atleast_2d(cls, X):
        import scipy.sparse as sp

        if sp.issparse(X):
            return X
        raise ValueError("LinalgSciPySparse requires sparse matrices")

    @classmethod
    def ceil(cls, X):
        import scipy.sparse as sp

        if sp.issparse(X):
            return X.ceil(X)
        raise ValueError("LinalgSciPySparse requires sparse matrices")

    @classmethod
    def cond(cls, X):
        import scipy.sparse as sp

        if sp.issparse(X):
            return sp.linalg.norm(X) * sp.linalg.norm(sp.linalg.inv(X))
        raise ValueError("LinalgSciPySparse requires sparse matrices")

    @classmethod
    def diag(cls, X):
        import scipy.sparse as sp

        if sp.issparse(X):
            return X.diagonal()
        raise ValueError("LinalgSciPySparse requires sparse matrices")

    @classmethod
    def eig(cls, X):
        import scipy.sparse as sp

        if sp.issparse(X):
            return sp.linalg.eig(X)

    @classmethod
    def full(cls, size, fill_value):
        raise ValueError(
            "full() is not recommended when dealing with sparse matrices"
        )

    @classmethod
    def inv(cls, X):
        import scipy as sp

        if sp.issparse(X):
            return sp.linalg.inv(X)

    @classmethod
    def log(cls, X):
        raise ValueError("log() is not recommended when dealing with sparse matrices")

    @classmethod
    def lstsq(cls, X, Y, rcond):
        import scipy.sparse as sp

        return sp.linalg.lsqr(X, Y, atol=rcond, btol=rcond)

    @classmethod
    def make_not_writeable(cls, X):
        # not supported
        logging.info(
            "SciPy.sparse does not support non-writeable tensors, ignoring ..."
        )

    @classmethod
    def median(cls, X, *args, **kwargs):
        import numpy as np

        return np.median(X)

    @classmethod
    def multi_dot(cls, Xs, *args, **kwargs):
        import numpy as np

        return np.linalg.multi_dot(Xs, *args, **kwargs)

    @classmethod
    def new_array(cls, X):
        import numpy as np

        return np.array(X)

    @classmethod
    def norm(cls, Xs, *args, **kwargs):
        import numpy as np

        return np.linalg.norm(X, *args, **kwargs)

    @classmethod
    def pow(cls, X, power):
        import numpy as np

        return np.power(X, power)

    @classmethod
    def repeat(cls, X, repeats, axis):
        import numpy as np

        return np.repeat(X, repeats, axis=axis)

    @classmethod
    def searchsorted(cls, X, val, *args, **kwargs):
        import numpy as np

        return np.searchsorted(X, val)

    @classmethod
    def solve(cls, A, b):
        import numpy as np

        return np.linalg.solve(A, b)

    @classmethod
    def sqrtm(cls, X):
        from scipy.linalg import sqrtm

        return sqrtm(X)

    @classmethod
    def svd(cls, X, *args, **kwargs):
        import numpy as np

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
        import numpy as np

        return np.vander(X, N, increasing)