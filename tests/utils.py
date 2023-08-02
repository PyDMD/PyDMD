from enum import Enum, auto
import numpy as np
import pytest
import torch

from pydmd.linalg import build_linalg_module

np.random.seed(10)


# Available backends for testing
class Backend(Enum):
    NUMPY = auto()
    PYTORCH_CPU = auto()


def fit_reconstruct(dmd):
    def func(X):
        batch = X.ndim == 3
        return dmd.fit(X, batch=batch).reconstructed_data

    return func


def numpyfy(X):
    if torch.is_tensor(X):
        return X.detach().resolve_conj().numpy()
    return np.array(X)


def assert_allclose(X, Y, *args, **kwargs):
    np.testing.assert_allclose(numpyfy(X), numpyfy(Y), *args, **kwargs)


def sample_data():
    # 15 snapshot with 400 data. The matrix is 400x15 and it contains
    # the following data: f1 + f2 where
    # f1 = lambda x,t: sech(x+3)*(1.*np.exp(1j*2.3*t))
    # f2 = lambda x,t: (sech(x)*np.tanh(x))*(2.*np.exp(1j*2.8*t))
    return np.load("tests/test_datasets/input_sample.npy")


def noisy_data():
    mu = 0.0
    sigma = 0.0  # noise standard deviation
    m = 100  # number of snapshot
    noise = np.random.normal(mu, sigma, m)  # gaussian noise
    A = np.array([[1.0, 1.0], [-1.0, 2.0]])
    A /= np.sqrt(3)
    n = 2
    X = np.zeros((n, m))
    X[:, 0] = np.array([0.5, 1.0])
    # evolve the system and perturb the data with noise
    for k in range(1, m):
        X[:, k] = A.dot(X[:, k - 1])
        X[:, k - 1] += noise[k - 1]
    return X


def setup_backends(data=None, exclude: Backend = None):
    if data is None:
        data = sample_data()

    if isinstance(data, dict):
        data_backends = {
            Backend.NUMPY: data,
            Backend.PYTORCH_CPU: {
                key: torch.from_numpy(arr) for key, arr in data.items()
            },
        }
    else:
        data_backends = {
            Backend.NUMPY: data,
            Backend.PYTORCH_CPU: torch.from_numpy(data),
        }

    if exclude is None:
        exclude = set()
    if isinstance(exclude, Backend):
        exclude = set((exclude,))
    return [
        pytest.param(arr, id=str(key))
        for key, arr in data_backends.items()
        if key not in exclude
    ]


def setup_linalg_module_backends(filters=None):
    # TODO: we expect things like `new_array` to go on GPU
    # automatically
    return [
        build_linalg_module(param.values[0])
        for param in setup_backends(data=None, filters=filters)
    ]
