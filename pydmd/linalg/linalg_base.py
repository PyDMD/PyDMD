class LinalgBase:
    def __init__(self):
        raise RuntimeError("Instances not allowed")

    @classmethod
    def abs(cls, X):
        raise NotImplementedError

    @classmethod
    def arange(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def atleast_1d(cls, X):
        raise NotImplementedError

    @classmethod
    def atleast_2d(cls, X):
        raise NotImplementedError

    @classmethod
    def argsort(cls, X, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def cat(cls, Xs, axis):
        raise NotImplementedError

    @classmethod
    def ceil(cls, X):
        raise NotImplementedError

    @classmethod
    def cond(cls, X):
        raise NotImplementedError

    @classmethod
    def device(cls, X):
        raise NotImplementedError

    @classmethod
    def diag_matrix(cls, X):
        raise NotImplementedError

    @classmethod
    def dot(cls, X, Y):
        raise NotImplementedError

    @classmethod
    def eig(cls, X):
        raise NotImplementedError

    @classmethod
    def extract_diagonal(cls, X):
        raise NotImplementedError

    @classmethod
    def full(cls, size, fill_value):
        raise NotImplementedError

    @classmethod
    def inv(cls, X):
        raise NotImplementedError

    @classmethod
    def isnan(cls, X):
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
    def multiply_elementwise(cls, X, Y):
        raise NotImplementedError

    @classmethod
    def multi_dot(cls, Xs, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def nanmean(cls, X, axis):
        raise NotImplementedError

    @classmethod
    def nansum(cls, X, axis):
        raise NotImplementedError

    @classmethod
    def new_array(cls, X):
        raise NotImplementedError

    @classmethod
    def norm(cls, Xs, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def pinv(cls, X):
        raise NotImplementedError

    @classmethod
    def pow(cls, X, power):
        raise NotImplementedError

    @classmethod
    def pseudo_hankel_matrix(cls, X, d):
        """
        Arrange the snapshot in the matrix `X` into the (pseudo) Hankel
        matrix. The attribute `d` controls the number of snapshot from `X` in
        each snapshot of the Hankel matrix.

        :Example:

            >>> a = np.array([[1, 2, 3, 4, 5]])
            >>> pseudo_hankel_matrix(a, 2)
            array([[1, 2, 3, 4],
                   [2, 3, 4, 5]])
            >>> pseudo_hankel_matrix(a, 4)
            array([[1, 2],
                   [2, 3],
                   [3, 4],
                   [4, 5]])

            >>> a = np.array([1,2,3,4,5,6]).reshape(2,3)
            >>> a
            array([[1, 2, 3],
                   [4, 5, 6]])
            >>> pseudo_hankel_matrix(a, 2)
            array([[1, 2],
                   [4, 5],
                   [2, 3],
                   [5, 6]])
        """
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
    def split(cls, X, n_arrays, axis):
        raise NotImplementedError

    @classmethod
    def matrix_sqrt(cls, X):
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


