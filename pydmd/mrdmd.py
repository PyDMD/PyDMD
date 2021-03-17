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
from .dmdoperator import DMDOperator
from .utils import compute_tlsq

class SubMrDMDOperator(DMDOperator):
    def __init__(self, step, rho, **kwargs):
        super(SubMrDMDOperator, self).__init__(**kwargs)

        self._step = step
        self._rho = rho

        self._slow_modes = None

    def compute_operator(self, Xc, Yc):
        U, s, V = self._compute_svd(Xc)

        self._Atilde = U.T.conj().dot(Yc).dot(V) * np.reciprocal(s)

        self._compute_eigenquantities()
        self._compute_modes_and_Lambda(Yc, U, s, V)

        self._slow_modes = (np.abs(old_div(np.log(self.eigenvalues),
            (2. * np.pi * self._step)))) <= self._rho

    def compute_sub_amplitudes(self, Xc, opt):
        if opt:
            # compute the vandermonde matrix
            omega = old_div(np.log(self.eigs), self.original_time['dt'])
            vander = np.exp(
                np.multiply(*np.meshgrid(omega, self.dmd_timesteps))).T

            # perform svd on all the snapshots
            U, s, V = np.linalg.svd(Xc, full_matrices=False)

            P = np.multiply(np.dot(self.modes.conj().T, self.modes),
                            np.conj(np.dot(vander,
                                           vander.conj().T)))
            tmp = (np.dot(np.dot(U, np.diag(s)), V)).conj().T
            q = np.conj(np.diag(np.dot(np.dot(vander, tmp), self.modes)))

            # b optimal
            a = np.linalg.solve(P, q)
        else:
            a = np.linalg.lstsq(self.modes, Xc.T[0], rcond=None)[0]

        return a

    @DMDOperator.modes.getter
    def modes(self):
        # we want to access eigenvalues before setting slow_modes, since
        # setting slow_modes requires evaluating eigenvalues
        if self._slow_modes is None:
            return super(SubMrDMDOperator, self).modes
        else:
            return super(SubMrDMDOperator, self).modes[:, self._slow_modes]

    @DMDOperator.eigenvalues.getter
    def eigenvalues(self):
        # we want to access eigenvalues before setting slow_modes, since
        # setting slow_modes requires evaluating eigenvalues
        if self._slow_modes is None:
            return super(SubMrDMDOperator, self).eigenvalues
        else:
            return super(SubMrDMDOperator, self).eigenvalues[self._slow_modes]

class MrDMDOperator(DMDOperator):
    def __init__(self, max_cycles=1, max_level=6, tlsq_rank=0, opt=False,
        **kwargs):
        super(MrDMDOperator, self).__init__(**kwargs)

        self._tlsq_rank = tlsq_rank
        self._max_cycles = max_cycles
        self._max_level = max_level
        self._opt = opt

        self._nyq = 8 * self._max_cycles

        # remember for the instances of SubMrDMDOperator
        self._args = kwargs

        # initialization
        self._modes = []
        self._b = []
        self._Atilde = []
        self._eigenvalues = []
        self._nsamples = []
        self._steps = []

    @DMDOperator.as_numpy_array.getter
    def as_numpy_array(self):
        raise RuntimeError("This property isn't defined")

    def compute_operator(self, data_queue):
        current_bin = 0
        while data_queue:
            Xraw = data_queue.pop(0)

            n_samples = Xraw.shape[1]

            step = max(1, int(np.floor(old_div(n_samples, self._nyq))))
            Xsub = Xraw[:, ::step]
            Xc = Xsub[:, :-1]
            Yc = Xsub[:, 1:]

            Xc, Yc = compute_tlsq(Xc, Yc, self._tlsq_rank)

            rho = old_div(float(self._max_cycles), n_samples)
            sub_operator = SubMrDMDOperator(step=step, rho=rho, **self._args)
            sub_operator.compute_operator(Xc, Yc)

            modes = sub_operator.modes
            eigs = sub_operator.eigenvalues
            Atilde = sub_operator.as_numpy_array
            b = sub_operator.compute_sub_amplitudes(Xc, self._opt)

            #---------------------------------------------------------------
            # DMD Amplitudes and Dynamics
            #---------------------------------------------------------------
            Vand = np.vander(np.power(eigs, old_div(1., step)), n_samples, True)

            Psi = (Vand.T * b).T

            self._modes.append(modes)
            self._b.append(b)
            self._Atilde.append(Atilde)
            self._eigenvalues.append(eigs)
            self._nsamples.append(n_samples)
            self._steps.append(step)

            if Xraw.dtype == 'float64':
                Xraw -= modes.dot(Psi).real
            else:
                Xraw -= modes.dot(Psi)

            if current_bin < 2**(self._max_level - 1) - 1:
                current_bin += 1
                half = int(np.ceil(old_div(Xraw.shape[1], 2)))
                data_queue.append(Xraw[:, :half])
                data_queue.append(Xraw[:, half:])
            else:
                current_bin += 1

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
        is 0, that means TLSQ is not applied.
    :param bool exact: flag to compute either exact DMD or projected DMD.
        Default is False.
    :param bool opt: flag to compute optimal amplitudes. See :class:`DMDBase`.
        Default is False.
    :param rescale_mode: Scale Atilde as shown in
            10.1016/j.jneumeth.2015.10.010 (section 2.4) before computing its
            eigendecomposition. None means no rescaling, 'auto' means automatic
            rescaling using singular values, otherwise the scaling factors.
    :type rescale_mode: {'auto'} or None or numpy.ndarray
    :param int max_cycles: the maximum number of mode oscillations in any given
        time scale. Default is 1.
    :param int max_level: the maximum number of levels. Defualt is 6.
    """

    def __init__(self, max_level=6, opt=False, tlsq_rank=0, **kwargs):
        self._initialize_dmdoperator(tlsq_rank=tlsq_rank, max_level=max_level, opt=opt, **kwargs)

        self._original_time = None
        self._dmd_time = None

        self._max_level = max_level

    def _initialize_dmdoperator(self, **kwargs):
        self._Atilde = MrDMDOperator(**kwargs)

    def _index_list(self, level, node):
        """
        Private method that return the right index element from a given level
        and node.

        :param int level: the level in the binary tree.
        :param int node: the node id.
        :rtype: int
        :return: the index of the list that contains the binary tree.
        """
        if level >= self._max_level:
            raise ValueError("Invalid level: greater than `max_level`")

        if node >= 2**level:
            raise ValueError("Invalid node")

        return 2**level + node - 1

    def _index_list_reversed(self, index):
        """
        Method that return the level and node given the index of the bin.

        :param int index: the index of the bin in the binary tree.
        :return: the level of the bin in the binary tree and the node id
            in that level.
        """
        if index > 2**self._max_level - 2:
            raise ValueError("Invalid index: maximum index is ({})".format(2**self._max_level - 2))
        for lvl in range(self._max_level + 1):
            if index < 2**lvl - 1:
                break
        level = lvl - 1
        node = index - 2**level + 1
        return level, node

    def partial_time_interval(self, level, node):
        """
        Evaluate the start and end time and the period of a given bin.

        :param int level: the level in the binary tree.
        :param int node: the node id.
        :return: the start and end time and the period of the bin
        :rtype: dictionary
        """
        if level >= self._max_level:
            raise ValueError(
                'The level input parameter ({}) has to be less than the '
                'max_level ({}). Remember that the starting index is 0'.format(
                    level, self._max_level))

        if node >= 2**level:
            raise ValueError("Invalid node")

        full_period = self.original_time['tend'] - self.original_time['t0']
        period = full_period / 2**level
        t0 = self.original_time['t0'] + period*node
        tend = t0 + period
        return {'t0': t0, 'tend':tend, 'dt':period}

    def time_window_bins(self, t0, tend):
        """
        Find which bins are embedded (partially or totally) in a given
        time window.

        :param float t0: start time of the window.
        :param float tend: end time of the window.
        :return: indexes of the bins seen by the time window.
        :rtype: numpy.ndarray
        """
        indexes = []
        for level in range(self._max_level):
            for i in range(2**level):
                local_times = self.partial_time_interval(level, i)
                if t0 >= local_times['t0'] and t0 < local_times['tend']:
                    indexes.append(self._index_list(level, i))
                if tend > local_times['t0'] and tend <= local_times['tend']:
                    indexes.append(self._index_list(level, i))
                if t0 <= local_times['t0'] and tend >= local_times['tend']:
                    indexes.append(self._index_list(level, i))
        # Remove duplicates if they exist
        # indexes = list(dict.fromkeys(indexes)) # Python 3.7 or later (preserve order)
        indexes = list(set(indexes)) # Any Python version, but does not preserve order
        indexes = np.sort(indexes)
        return indexes

    def time_window_eigs(self, t0, tend):
        """
        Get the eigenvalues relative to the modes of the bins embedded (partially
        or totally) in a given time window.

        :param float t0: start time of the window.
        :param float tend: end time of the window.
        :return: the eigenvalues for that time window.
        :rtype: numpy.ndarray
        """
        indexes = self.time_window_bins(t0, tend)
        return np.concatenate([self._Atilde.eigenvalues[idx] for idx in indexes])

    def time_window_frequency(self, t0, tend):
        """
        Get the frequencies relative to the modes of the bins embedded (partially
        or totally) in a given time window.

        :param float t0: start time of the window.
        :param float tend: end time of the window.
        :return: the frequencies for that time window.
        :rtype: numpy.ndarray
        """
        eigs = self.time_window_eigs(t0, tend)
        return np.log(eigs).imag/(2*np.pi*self.original_time['dt'])

    def time_window_growth_rate(self, t0, tend):
        """
        Get the growth rate values relative to the modes of the bins embedded (partially
        or totally) in a given time window.

        :param float t0: start time of the window.
        :param float tend: end time of the window.
        :return: the Floquet values for that time window.
        :rtype: numpy.ndarray
        """
        return self.time_window_eigs(t0, tend).real/self.original_time['dt']

    def time_window_amplitudes(self, t0, tend):
        """
        Get the amplitudes relative to the modes of the bins embedded (partially
        or totally) in a given time window.

        :param float t0: start time of the window.
        :param float tend: end time of the window.
        :return: the amplitude of the modes for that time window.
        :rtype: numpy.ndarray
        """
        indexes = self.time_window_bins(t0, tend)
        return np.concatenate([self._b[idx] for idx in indexes])

    @property
    def reconstructed_data(self):
        """
        Get the reconstructed data.

        :return: the matrix that contains the reconstructed snapshots.
        :rtype: numpy.ndarray
        """
        try:
            data = np.sum(
                np.array([
                    self.partial_reconstructed_data(i)
                    for i in range(self._max_level)
                ]),
                axis=0)
        except MemoryError:
            data = np.array(self.partial_reconstructed_data(0))
            for i in range(1, self._max_level):
                data = np.sum([data,
                np.array(self.partial_reconstructed_data(i))], axis=0)
        return data

    @property
    def modes(self):
        """
        Get the matrix containing the DMD modes, stored by column.

        :return: the matrix containing the DMD modes.
        :rtype: numpy.ndarray
        """
        return np.hstack(tuple(self._Atilde.modes))

    @property
    def dynamics(self):
        """
        Get the time evolution of each mode.

        :return: the matrix that contains all the time evolution, stored by
                row.
        :rtype: numpy.ndarray
        """
        return np.vstack(
            tuple([self.partial_dynamics(i) for i in range(self._max_level)]))

    @property
    def eigs(self):
        """
        Get the eigenvalues of A tilde.

        :return: the eigenvalues from the eigendecomposition of `atilde`.
        :rtype: numpy.ndarray
        """
        return np.concatenate(self._Atilde.eigenvalues)

    @property
    def _steps(self):
        return self._Atilde._steps

    @property
    def _nsamples(self):
        return self._Atilde._nsamples

    @property
    def _b(self):
        return self._Atilde._b

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
            return self._Atilde.modes[self._index_list(level, node)]

        indeces = [self._index_list(level, i) for i in range(2**level)]
        return np.hstack(tuple([self._Atilde.modes[idx] for idx in indeces]))

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
            dynamic(self._Atilde.eigenvalues[idx], self._b[idx], self._steps[idx],
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
        if level >= self._max_level:
            raise ValueError(
                'The level input parameter ({}) has to be less than the'
                'max_level ({}). Remember that the starting index is 0'.format(
                    level, self._max_level))
        if node:
            return self._Atilde.eigenvalues[self._index_list(level, node)]

        indeces = [self._index_list(level, i) for i in range(2**level)]
        return np.concatenate([self._Atilde.eigenvalues[idx] for idx in indeces])

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
        if level >= self._max_level:
            raise ValueError(
                'The level input parameter ({}) has to be less than the '
                'max_level ({}). Remember that the starting index is 0'.format(
                    level, self._max_level))
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

        # Redefine max level if it is too big.
        lvl_threshold = int(np.log(self._snapshots.shape[1]/4.)/np.log(2.)) + 1
        if self._max_level > lvl_threshold:
            self._max_level = lvl_threshold
            print('Too many levels... '
                  'Redefining `max_level` to {}'.format(self._max_level))

        self._Atilde.compute_operator(data_queue)

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
        if self._Atilde.eigenvalues is None:
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
            colors = [cmap(i) for i in np.linspace(0, 1, self._max_level)]

            points = []
            for lvl in range(self._max_level):
                indeces = [self._index_list(lvl, i) for i in range(2 ** lvl)]
                eigs = np.concatenate([self._Atilde.eigenvalues[idx] for idx in indeces])

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
                for i in range(self._max_level)
            ]

        if show_unit_circle:
            points += [unit_circle]
            labels += ['Unit circle']

        ax.add_artist(plt.legend(points, labels, loc='best'))
        plt.show()
