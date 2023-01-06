import numpy as np
import pytest
import torch

np.random.seed(10)


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


def setup_backends(data=None, filters=None):
    if data is None:
        data = sample_data()

    if isinstance(data, dict):
        data_backends = {
            "NumPy": data,
            "PyTorch CPU": {
                key: torch.from_numpy(arr) for key, arr in data.items()
            },
        }
    else:
        data_backends = {
            "NumPy": data,
            "PyTorch CPU": torch.from_numpy(data)
        }

    if filters is None:
        filters = set()
    return [
        pytest.param(arr, id=key)
        for key, arr in data_backends.items()
        if key not in filters
    ]
