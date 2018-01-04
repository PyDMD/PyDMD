"""
Derived module from dmdbase.py for multi-resolution dmd.

Reference:
- Kutz, J. Nathan, Xing Fu, and Steven L. Brunton. Multiresolution Dynamic Mode
Decomposition. SIAM Journal on Applied Dynamical Systems 15.2 (2016): 713-735.
"""
from __future__ import division
from builtins import range
from past.utils import old_div
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

from .dmdbase import DMDBase


class MrDMD(DMDBase):
    """
    Multi-resolution Dynamic Mode Decomposition

    :param svd_rank: the rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive interger, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation.
    :type svd_rank: int or float
    :param int tlsq_rank: rank truncation computing Total Least Square. Default
        is 0, that means no truncation.
    :param bool exact: flag to compute either exact DMD or projected DMD.
        Default is False.
    :param bool opt: flag to compute optimized DMD. Default is False.
    :param int max_cycles: the maximum number of mode oscillations in any given
        time scale. Default is 1.
    :param int max_level: the maximum number of levels. Defualt is 6.
    """

    def __init__(self,
                 svd_rank=0,
                 tlsq_rank=0,
                 exact=False,
                 opt=False,
                 max_cycles=1,
                 max_level=6):
        super(MrDMD, self).__init__(svd_rank, tlsq_rank, exact, opt)
        self.max_cycles = max_cycles
        self.max_level = max_level
        self._nsamples = None
        self._steps = None

    def _index_list(self, level, node):
        """
        Private method that return the right index element from a given level
        and node.

        :param int level: the level in the binary tree.
        :param int node: the node id.
        :rtype: int
        :return: the index of the list that contains the binary tree.
        """
        if level > self.max_level:
            raise ValueError("Invalid level: greater than `max_level`")

        if node >= 2**level:
            raise ValueError("Invalid node")

        return 2**level + node - 1

    @property
    def reconstructed_data(self):
        """
        Get the reconstructed data.

        :return: the matrix that contains the reconstructed snapshots.
        :rtype: numpy.ndarray
        """
        data = np.sum(
            np.array([
                self.partial_reconstructed_data(i)
                for i in range(self.max_level)
            ]),
            axis=0)
        return data

    @property
    def modes(self):
        """
        Get the matrix containing the DMD modes, stored by column.

        :return: the matrix containing the DMD modes.
        :rtype: numpy.ndarray
        """
        return np.hstack(tuple(self._modes))

    @property
    def dynamics(self):
        """
        Get the time evolution of each mode.

        :return: the matrix that contains all the time evolution, stored by
                row.
        :rtype: numpy.ndarray
        """
        return np.vstack(
            tuple([self.partial_dynamics(i) for i in range(self.max_level)]))

    @property
    def eigs(self):
        """
        Get the eigenvalues of A tilde.

        :return: the eigenvalues from the eigendecomposition of `atilde`.
        :rtype: numpy.ndarray
        """
        return np.concatenate(self._eigs)

    def partial_modes(self, level, node=None):
        """
        Return the modes at the specific `level` and at the specific `node`; if
        `node` is not specified, the method returns all the modes of the given
        `level` (all the nodes).

        :param int level: the index of the level from where the modes are
            extracted.
        :param int node: the index of the node from where the modes are
            extracted; if None, the modes are extracted from all the nodes of
            the given level. Default is None.
        """
        if node:
            return self._modes[self._index_list(level, node)]

        indeces = [self._index_list(level, i) for i in range(2**level)]
        return np.hstack(tuple([self._modes[idx] for idx in indeces]))

    def partial_dynamics(self, level, node=None):
        """
        Return the time evolution of the specific `level` and of the specific
        `node`; if `node` is not specified, the method returns the time
        evolution of the given `level` (all the nodes).

        :param int level: the index of the level from where the time evolution
            is extracted.
        :param int node: the index of the node from where the time evolution is
            extracted; if None, the time evolution is extracted from all the
            nodes of the given level. Default is None.
        """

        def dynamic(eigs, amplitudes, step, nsamples):
            omega = old_div(
                np.log(np.power(eigs, old_div(1., step))),
                self.original_time['dt'])
            partial_timestep = np.arange(nsamples) * self.dmd_time['dt']
            vander = np.exp(np.multiply(*np.meshgrid(omega, partial_timestep)))
            return (vander * amplitudes).T

        if node:
            indeces = [self._index_list(level, node)]
        else:
            indeces = [self._index_list(level, i) for i in range(2**level)]

        level_dynamics = [
            dynamic(self._eigs[idx], self._b[idx], self._steps[idx],
                    self._nsamples[idx]) for idx in indeces
        ]
        return scipy.linalg.block_diag(*level_dynamics)

    def partial_eigs(self, level, node=None):
        """
        Return the eigenvalues of the specific `level` and of the specific
        `node`; if `node` is not specified, the method returns the eigenvalues
        of the given `level` (all the nodes).

        :param int level: the index of the level from where the eigenvalues is
            extracted.
        :param int node: the index of the node from where the eigenvalues is
            extracted; if None, the time evolution is extracted from all the
            nodes of the given level. Default is None.
        """
        if level >= self.max_level:
            raise ValueError(
                'The level input parameter ({}) has to be less then the'
                'max_level ({}). Remember that the starting index is 0'.format(
                    level, self.max_level))
        if node:
            return self._eigs[self._index_list(level, node)]

        indeces = [self._index_list(level, i) for i in range(2**level)]
        return np.concatenate([self._eigs[idx] for idx in indeces])

    def partial_reconstructed_data(self, level, node=None):
        """
        Return the reconstructed data computed using the modes and the time
        evolution at the specific `level` and at the specific `node`; if `node`
        is not specified, the method returns the reconstructed data
        of the given `level` (all the nodes).

        :param int level: the index of the level.
        :param int node: the index of the node from where the time evolution is
            extracted; if None, the time evolution is extracted from all the
            nodes of the given level. Default is None.

        """
        if level >= self.max_level:
            raise ValueError(
                'The level input parameter ({}) has to be less then the'
                'max_level ({}). Remember that the starting index is 0'.format(
                    level, self.max_level))
        modes = self.partial_modes(level, node)
        dynamics = self.partial_dynamics(level, node)

        return modes.dot(dynamics)

    def fit(self, X):
        """
        Compute the Dynamic Modes Decomposition to the input data.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        """
        self._snapshots, self._snapshots_shape = self._col_major_2darray(X)

        # To avoid recursion function, use FIFO list to simulate the tree
        # structure
        data_queue = [self._snapshots.copy()]

        current_level = 1

        # Reset the lists
        self._eigs = []
        self._Atilde = []
        self._modes = []
        self._b = []
        self._nsamples = []
        self._steps = []

        while data_queue:
            Xraw = data_queue.pop(0)

            n_samples = Xraw.shape[1]
            # subsamples frequency to detect slow modes
            nyq = 8 * self.max_cycles

            try:
                step = int(np.floor(old_div(n_samples, nyq)))
                Xsub = Xraw[:, ::step]
                Xc = Xsub[:, :-1]
                Yc = Xsub[:, 1:]

                Xc, Yc = self._compute_tlsq(Xc, Yc, self.tlsq_rank)

                U, s, V = self._compute_svd(Xc, self.svd_rank)

                Atilde = self._build_lowrank_op(U, s, V, Yc)

                eigs, modes = self._eig_from_lowrank_op(Atilde, Yc, U, s, V,
                                                        self.exact)
                rho = old_div(float(self.max_cycles), n_samples)
                slow_modes = (np.abs(
                    old_div(np.log(eigs), (2. * np.pi * step)))) <= rho

                modes = modes[:, slow_modes]
                eigs = eigs[slow_modes]

                #---------------------------------------------------------------
                # DMD Amplitudes and Dynamics
                #---------------------------------------------------------------
                Vand = np.vander(
                    np.power(eigs, old_div(1., step)), n_samples, True)
                b = self._compute_amplitudes(modes, Xc, eigs, self.opt)

                Psi = (Vand.T * b).T
            except:
                modes = np.zeros((Xraw.shape[0], 1))
                Psi = np.zeros((1, Xraw.shape[1]))
                Atilde = np.zeros(0)
                eigs = np.zeros(0)
                b = np.zeros(0)
                step = 0

            self._modes.append(modes)
            self._b.append(b)
            self._Atilde.append(Atilde)
            self._eigs.append(eigs)
            self._nsamples.append(n_samples)
            self._steps.append(step)

            Xraw -= modes.dot(Psi)

            if current_level < 2**(self.max_level - 1):
                current_level += 1
                half = int(np.ceil(old_div(Xraw.shape[1], 2)))
                data_queue.append(Xraw[:, :half])
                data_queue.append(Xraw[:, half:])

        self.dmd_time = {'t0': 0, 'tend': self._snapshots.shape[1], 'dt': 1}
        self.original_time = self.dmd_time.copy()

        return self

    def plot_eigs(self,
                  show_axes=True,
                  show_unit_circle=True,
                  figsize=(8, 8),
                  title='',
                  level=None,
                  node=None):
        """
        Plot the eigenvalues.

        :param bool show_axes: if True, the axes will be showed in the plot.
                Default is True.
        :param bool show_unit_circle: if True, the circle with unitary radius
                and center in the origin will be showed. Default is True.
        :param tuple(int,int) figsize: tuple in inches of the figure.
        :param str title: title of the plot.
        :param int level: plot only the eigenvalues of specific level.
        :param int node: plot only the eigenvalues of specific node.
        """
        if self._eigs is None:
            raise ValueError('The eigenvalues have not been computed.'
                             'You have to perform the fit method.')

        if level:
            peigs = self.partial_eigs(level=level, node=node)
        else:
            peigs = self.eigs

        plt.figure(figsize=figsize)
        plt.title(title)
        plt.gcf()
        ax = plt.gca()

        if not level:
            cmap = plt.get_cmap('viridis')
            colors = [cmap(i) for i in np.linspace(0, 1, self.max_level)]

            points = []
            for lvl in range(self.max_level):
                indeces = [self._index_list(lvl, i) for i in range(2**lvl)]
                eigs = np.concatenate([self._eigs[idx] for idx in indeces])

                points.append(
                    ax.plot(eigs.real, eigs.imag, '.', color=colors[lvl])[0])
        else:
            points = []
            points.append(
                ax.plot(peigs.real, peigs.imag, 'bo', label='Eigenvalues')[0])

        # set limits for axis
        limit = np.max(np.ceil(np.absolute(peigs)))
        ax.set_xlim((-limit, limit))
        ax.set_ylim((-limit, limit))

        plt.ylabel('Imaginary part')
        plt.xlabel('Real part')

        if show_unit_circle:
            unit_circle = plt.Circle(
                (0., 0.), 1., color='green', fill=False, linestyle='--')
            ax.add_artist(unit_circle)

        # Dashed grid
        gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        for line in gridlines:
            line.set_linestyle('-.')
        ax.grid(True)

        ax.set_aspect('equal')

        # x and y axes
        if show_axes:
            ax.annotate(
                '',
                xy=(np.max([limit * 0.8, 1.]), 0.),
                xytext=(np.min([-limit * 0.8, -1.]), 0.),
                arrowprops=dict(arrowstyle="->"))
            ax.annotate(
                '',
                xy=(0., np.max([limit * 0.8, 1.])),
                xytext=(0., np.min([-limit * 0.8, -1.])),
                arrowprops=dict(arrowstyle="->"))

        # legend
        if level:
            labels = ['Eigenvalues - level {}'.format(level)]
        else:
            labels = [
                'Eigenvalues - level {}'.format(i)
                for i in range(self.max_level)
            ]

        if show_unit_circle:
            points += [unit_circle]
            labels += ['Unit circle']

        ax.add_artist(plt.legend(points, labels, loc='best'))
        plt.show()
