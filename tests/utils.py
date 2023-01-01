import numpy as np
import torch
import pytest

def assert_allclose(X, Y, *args, **kwargs):
    X = np.array(X)
    Y = np.array(Y)
    np.testing.assert_allclose(X, Y, *args, **kwargs)

def load_sample_data():
    # 15 snapshot with 400 data. The matrix is 400x15 and it contains
    # the following data: f1 + f2 where
    # f1 = lambda x,t: sech(x+3)*(1.*np.exp(1j*2.3*t))
    # f2 = lambda x,t: (sech(x)*np.tanh(x))*(2.*np.exp(1j*2.8*t))
    return np.load('tests/test_datasets/input_sample.npy')

sample_data = load_sample_data()
data_backends = (
    # NumPy
    pytest.param(sample_data, id="NumPy"),
    # PyTorch
    pytest.param(torch.from_numpy(sample_data), id="PyTorch CPU"),
)