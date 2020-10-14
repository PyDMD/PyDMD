import matplotlib.pyplot as plt
import numpy as np
import time

from pydmd import FbDMD
from pydmd import DMD

from ipywidgets import interact_manual, interact, interactive, FloatSlider

snapshots = [
    np.genfromtxt('data/velocity0.{}.csv'.format(i), delimiter=',', skip_header=1)[:, 0]
    for i in range(20, 40)
]

pts = np.genfromtxt('data/velocity0.20.csv', delimiter=',', skip_header=1)[:, -3:-1]


plt.figure(figsize=(16, 16))
for i, snapshot in enumerate(snapshots[::5], start=1):
    plt.subplot(2, 2, i)
    plt.scatter(pts[:, 0], pts[:, 1], c=snapshot, marker='.')
plt.show()

fbdmd = FbDMD(exact=True)
fbdmd.fit(snapshots)
fbdmd.reconstructed_data.shape


dmd = DMD(exact=True)
dmd.fit(snapshots)

print('[DMD  ] Total distance between eigenvalues and unit circle: {}'.format(
    np.sum(np.abs(dmd.eigs.real**2 + dmd.eigs.imag**2 - 1))
))
print('[FbDMD] Total distance between eigenvalues and unit circle: {}'.format(
    np.sum(np.abs(fbdmd.eigs.real**2 + fbdmd.eigs.imag**2 - 1))
))

dmd.plot_eigs()
fbdmd.plot_eigs()


fbdmd.dmd_time['dt'] *= .5
fbdmd.dmd_time['tend'] += 10

plt.plot(fbdmd.dmd_timesteps, fbdmd.dynamics.T.real)
plt.show()


def plot_state(time):
    i = int((time - fbdmd.dmd_time['t0']) / fbdmd.dmd_time['dt'])
    plt.figure(figsize=(8, 8))
    plt.scatter(pts[:, 0], pts[:, 1], c=fbdmd.reconstructed_data[:, i].real, marker='.')
    plt.show()


interactive_plot = interactive(
    plot_state,
    time=FloatSlider(
        min=fbdmd.dmd_time['t0'],
        max=fbdmd.dmd_time['tend'],
        step=fbdmd.dmd_time['dt'],
        value=0,
        continuous_update=False
    )
)
output = interactive_plot.children[-1]
output.layout.height = '500px'
interactive_plot
