class LinalgBase:
    @classmethod
    def abs(cls, X):
        raise NotImplementedError

    @classmethod
    def append(cls, X, Y, axis):
        raise NotImplementedError

    @classmethod
    def arange(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def atleast_2d(cls, X):
        raise NotImplementedError

    @classmethod
    def ceil(cls, X):
        raise NotImplementedError

    def cond(cls, X):
        raise NotImplementedError

    @classmethod
    def diag(cls, X):
        raise NotImplementedError

    @classmethod
    def eig(cls, X):
        raise NotImplementedError

    @classmethod
    def full(cls, size, fill_value):
        raise NotImplementedError

    @classmethod
    def inv(cls, X):
        raise NotImplementedError

    @classmethod
    def log(cls, X):
        raise NotImplementedError

    @classmethod
    def lstsq(cls, X, Y, rcond):
        raise NotImplementedError

    @classmethod
    def make_not_writeable(cls, X):
        raise NotImplementedError

    @classmethod
    def median(cls, X, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def multi_dot(cls, Xs, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def new_array(cls, X):
        raise NotImplementedError

    @classmethod
    def norm(cls, Xs, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def pow(cls, X, power):
        raise NotImplementedError

    @classmethod
    def repeat(cls, X, repeats, axis):
        raise NotImplementedError

    @classmethod
    def searchsorted(cls, X, val, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def solve(cls, A, b):
        raise NotImplementedError

    @classmethod
    def sqrtm(cls, X):
        raise NotImplementedError

    @classmethod
    def svd(cls, X, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def to(cls, X, other):
        raise NotImplementedError

    @classmethod
    def vander(cls, X, N, increasing):
        raise NotImplementedError