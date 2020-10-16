import matplotlib.pyplot as plt
import numpy as np
import scipy

from pydmd import DMDc


def create_system(n, m):
    A = scipy.linalg.helmert(n, True)
    B = np.random.rand(n, n)-.5
    x0 = np.array([0.25]*n)
    u = np.random.rand(n, m-1)-.5
    snapshots = [x0]
    for i in range(m-1):
        snapshots.append(A.dot(snapshots[i])+B.dot(u[:, i]))
    snapshots = np.array(snapshots).T
    return {'snapshots': snapshots, 'u': u, 'B': B, 'A': A}


s = create_system(25, 10)
print(s['snapshots'].shape)

dmdc = DMDc(svd_rank=-1)
dmdc.fit(s['snapshots'], s['u'])

plt.figure(figsize=(16, 6))
plt.subplot(121)
plt.title('Original system')
plt.pcolor(s['snapshots'].real)
plt.colorbar()

plt.subplot(122)
plt.title('Reconstructed system')
plt.pcolor(dmdc.reconstructed_data().real)
plt.colorbar()
plt.show()

new_u = np.exp(s['u'])

plt.figure(figsize=(8, 6))
plt.pcolor(dmdc.reconstructed_data(new_u).real)
plt.colorbar()
plt.show()

dmdc.dmd_time['dt'] = .5
new_u = np.random.rand(s['u'].shape[0], dmdc.dynamics.shape[1]-1)

plt.figure(figsize=(8, 6))
plt.pcolor(dmdc.reconstructed_data(new_u).real)
plt.colorbar()
plt.show()
