from unittest import TestCase
from pydmd.dmdbase import DMDBase
import matplotlib.pyplot as plt
import numpy as np

# 15 snapshot with 400 data. The matrix is 400x15 and it contains
# the following data: f1 + f2 where
# f1 = lambda x,t: sech(x+3)*(1.*np.exp(1j*2.3*t))
# f2 = lambda x,t: (sech(x)*np.tanh(x))*(2.*np.exp(1j*2.8*t))
sample_data = np.load('tests/test_datasets/input_sample.npy')


class TestDmdBase(TestCase):
    def test_svd_rank_default(self):
        dmd = DMDBase()
        assert dmd._Atilde._svd_rank == 0

    def test_svd_rank(self):
        dmd = DMDBase(svd_rank=3)
        assert dmd._Atilde._svd_rank == 3

    def test_tlsq_rank_default(self):
        dmd = DMDBase()
        assert dmd.tlsq_rank == 0

    def test_tlsq_rank(self):
        dmd = DMDBase(tlsq_rank=2)
        assert dmd.tlsq_rank == 2

    def test_exact_default(self):
        dmd = DMDBase()
        assert dmd._Atilde._exact == False

    def test_exact(self):
        dmd = DMDBase(exact=True)
        assert dmd._Atilde._exact == True

    def test_opt_default(self):
        dmd = DMDBase()
        assert dmd.opt == False

    def test_opt(self):
        dmd = DMDBase(opt=True)
        assert dmd.opt == True

    def test_fit(self):
        dmd = DMDBase(exact=False)
        with self.assertRaises(NotImplementedError):
            dmd.fit(sample_data)

    def test_plot_eigs(self):
        dmd = DMDBase()
        with self.assertRaises(ValueError):
            dmd.plot_eigs(show_axes=True, show_unit_circle=True)

    def test_plot_modes_2D(self):
        dmd = DMDBase()
        with self.assertRaises(ValueError):
            dmd.plot_modes_2D()

    def test_plot_snaps_2D(self):
        dmd = DMDBase()
        with self.assertRaises(ValueError):
            dmd.plot_snapshots_2D()
