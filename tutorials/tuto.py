# --> Import standard python packages.
import numpy as np

# --> Import matplotlib related.
import matplotlib.pyplot as plt

# -->
from pydmd import optDMD

def f1(x, t):
    return 1.0/np.cosh(x+3.0) * np.exp(2.3j*t)

def f2(x, t):
    return 2.0/np.cosh(x) * np.tanh(x) * np.exp(2.8j*t)


if __name__ == "__main__":

    x, t = np.linspace(-5, 5, 128), np.linspace(0, 4*np.pi, 256)

    xgrid, tgrid = np.meshgrid(x, t)

    X1, X2 = f1(xgrid, tgrid), f2(xgrid, tgrid)
    X = X1 + X2

    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)

    cmap = plt.cm.get_cmap("RdBu")

    axes[0].pcolormesh(
        xgrid, tgrid, X.real,
        shading="gouraud",
        cmap=cmap)

    axes[1].pcolormesh(
        xgrid, tgrid, X1.real,
        shading="gouraud",
        cmap=cmap)

    axes[2].pcolormesh(
        xgrid, tgrid, X2.real,
        shading="gouraud",
        cmap=cmap)


    dmd = optDMD(rank=2, factorization="evd")
    L, R, S = dmd.fit(X.T)

    fig, axes = plt.subplots(R.shape[1], 1, sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(R[:, i].real)
        ax.plot(R[:, i].imag)

    plt.show()
