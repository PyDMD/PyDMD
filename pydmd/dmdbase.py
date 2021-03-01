"""
Base module for the DMD: `fit` method must be implemented in inherited classes
"""
from __future__ import division

import warnings
from builtins import object
from builtins import range
from os.path import splitext

import matplotlib as mpl
import numpy as np
from past.utils import old_div

mpl.rcParams['figure.max_open_warning'] = 0
import matplotlib.pyplot as plt


class DMDBase(object):
    """
    Dynamic Mode Decomposition base class.

    :param int svd_rank: rank truncation in SVD. If 0, the method computes the
        optimal rank and uses it for truncation; if positive number, the method
        uses the argument for the truncation; if -1, the method does not
        compute truncation.
    :param int tlsq_rank: rank truncation computing Total Least Square. Default
        is 0, that means no truncation.
    :param bool exact: flag to compute either exact DMD or projected DMD.
        Default is False.
    :param bool opt: flag to compute optimized DMD. Default is False.
    :param numpy.array rescale_mode: None means no rescaling, 'auto' means
        automatic rescaling using SV, otherwise the user chooses the preferred
        scaling.
    :cvar dict original_time: dictionary that contains information about the
        time window where the system is sampled:

           - `t0` is the time of the first input snapshot;
           - `tend` is the time of the last input snapshot;
           - `dt` is the delta time between the snapshots.

    :cvar dict dmd_time: dictionary that contains information about the time
        window where the system is reconstructed:

            - `t0` is the time of the first approximated solution;
            - `tend` is the time of the last approximated solution;
            - `dt` is the delta time between the approximated solutions.

    """

    def __init__(self, svd_rank=0, tlsq_rank=0, exact=False, opt=False,
        rescale_mode=None):
        self.rescale_mode = rescale_mode
        self.svd_rank = svd_rank
        self.tlsq_rank = tlsq_rank
        self.exact = exact
        self.opt = opt
        self.original_time = None
        self.dmd_time = None

        self._eigs = None
        self._Atilde = None
        self._modes = None  # Phi
        self._b = None  # amplitudes
        self._snapshots = None
        self._snapshots_shape = None

    @property
    def dmd_timesteps(self):
        """
        Get the timesteps of the reconstructed states.

        :return: the time intervals of the original snapshots.
        :rtype: numpy.ndarray
        """
        return np.arange(self.dmd_time['t0'],
                         self.dmd_time['tend'] + self.dmd_time['dt'],
                         self.dmd_time['dt'])

    @property
    def original_timesteps(self):
        """
        Get the timesteps of the original snapshot.

        :return: the time intervals of the original snapshots.
        :rtype: numpy.ndarray
        """
        return np.arange(self.original_time['t0'],
                         self.original_time['tend'] + self.original_time['dt'],
                         self.original_time['dt'])

    @property
    def modes(self):
        """
        Get the matrix containing the DMD modes, stored by column.

        :return: the matrix containing the DMD modes.
        :rtype: numpy.ndarray
        """
        return self._modes

    @property
    def atilde(self):
        """
        Get the reduced Koopman operator A, called A tilde.

        :return: the reduced Koopman operator A.
        :rtype: numpy.ndarray
        """
        return self._Atilde

    @property
    def eigs(self):
        """
        Get the eigenvalues of A tilde.

        :return: the eigenvalues from the eigendecomposition of `atilde`.
        :rtype: numpy.ndarray
        """
        return self._eigs

    @property
    def dynamics(self):
        """
        Get the time evolution of each mode.

        :return: the matrix that contains all the time evolution, stored by
            row.
        :rtype: numpy.ndarray
        """
        omega = old_div(np.log(self.eigs), self.original_time['dt'])
        vander = np.exp(
            np.outer(omega, self.dmd_timesteps - self.original_time['t0']))
        return vander * self._b[:, None]

    @property
    def reconstructed_data(self):
        """
        Get the reconstructed data.

        :return: the matrix that contains the reconstructed snapshots.
        :rtype: numpy.ndarray
        """
        return self.modes.dot(self.dynamics)

    @property
    def snapshots(self):
        """
        Get the original input data.

        :return: the matrix that contains the original snapshots.
        :rtype: numpy.ndarray
        """
        return self._snapshots

    @property
    def frequency(self):
        """
        Get the amplitude spectrum.

        :return: the array that contains the frequencies of the eigenvalues.
        :rtype: numpy.ndarray
        """
        return np.log(self.eigs).imag / (2 * np.pi * self.original_time['dt'])

    @property
    def amplitudes(self):
        """
        Get the coefficients that minimize the error between the original
        system and the reconstructed one. For futher information, see
        `dmdbase._compute_amplitudes`.

        :return: the array that contains the amplitudes coefficient.
        :rtype: numpy.ndarray
        """
        return self._b

    def fit(self, X):
        """
        Abstract method to fit the snapshots matrices.

        Not implemented, it has to be implemented in subclasses.
        """
        raise NotImplementedError(
            'Subclass must implement abstract method {}.fit'.format(
                self.__class__.__name__))

    @staticmethod
    def _col_major_2darray(X):
        """
        Private method that takes as input the snapshots and stores them into a
        2D matrix, by column. If the input data is already formatted as 2D
        array, the method saves it, otherwise it also saves the original
        snapshots shape and reshapes the snapshots.

        :param X: the input snapshots.
        :type X: int or numpy.ndarray
        :return: the 2D matrix that contains the flatten snapshots, the shape
            of original snapshots.
        :rtype: numpy.ndarray, tuple
        """
        # If the data is already 2D ndarray
        if isinstance(X, np.ndarray) and X.ndim == 2:
            snapshots = X
            snapshots_shape = None
        else:
            input_shapes = [np.asarray(x).shape for x in X]

            if len(set(input_shapes)) != 1:
                raise ValueError('Snapshots have not the same dimension.')

            snapshots_shape = input_shapes[0]
            snapshots = np.transpose([np.asarray(x).flatten() for x in X])

        # check condition number of the data passed in
        cond_number = np.linalg.cond(snapshots)
        if cond_number > 10e4:
            warnings.warn("Input data matrix X has condition number {}. "
                          "Consider preprocessing data, passing in augmented data matrix, or regularization methods."
                          .format(cond_number))

        return snapshots, snapshots_shape

    @staticmethod
    def _compute_tlsq(X, Y, tlsq_rank):
        """
        Compute Total Least Square.

        :param numpy.ndarray X: the first matrix;
        :param numpy.ndarray Y: the second matrix;
        :param int tlsq_rank: the rank for the truncation; If 0, the method
            does not compute any noise reduction; if positive number, the
            method uses the argument for the SVD truncation used in the TLSQ
            method.
        :return: the denoised matrix X, the denoised matrix Y
        :rtype: numpy.ndarray, numpy.ndarray

        References:
        https://arxiv.org/pdf/1703.11004.pdf
        https://arxiv.org/pdf/1502.03854.pdf
        """
        # Do not perform tlsq
        if tlsq_rank == 0:
            return X, Y

        V = np.linalg.svd(np.append(X, Y, axis=0), full_matrices=False)[-1]
        rank = min(tlsq_rank, V.shape[0])
        VV = V[:rank, :].conj().T.dot(V[:rank, :])

        return X.dot(VV), Y.dot(VV)

    @staticmethod
    def _compute_svd(X, svd_rank):
        """
        Truncated Singular Value Decomposition.

        :param numpy.ndarray X: the matrix to decompose.
        :param svd_rank: the rank for the truncation; If 0, the method computes
            the optimal rank and uses it for truncation; if positive interger,
            the method uses the argument for the truncation; if float between 0
            and 1, the rank is the number of the biggest singular values that
            are needed to reach the 'energy' specified by `svd_rank`; if -1,
            the method does not compute truncation.
        :type svd_rank: int or float
        :return: the truncated left-singular vectors matrix, the truncated
            singular values array, the truncated right-singular vectors matrix.
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray

        References:
        Gavish, Matan, and David L. Donoho, The optimal hard threshold for
        singular values is, IEEE Transactions on Information Theory 60.8
        (2014): 5040-5053.
        """
        U, s, V = np.linalg.svd(X, full_matrices=False)
        V = V.conj().T

        if svd_rank == 0:
            omega = lambda x: 0.56 * x ** 3 - 0.95 * x ** 2 + 1.82 * x + 1.43
            beta = np.divide(*sorted(X.shape))
            tau = np.median(s) * omega(beta)
            rank = np.sum(s > tau)
        elif svd_rank > 0 and svd_rank < 1:
            cumulative_energy = np.cumsum(s ** 2 / (s ** 2).sum())
            rank = np.searchsorted(cumulative_energy, svd_rank) + 1
        elif svd_rank >= 1 and isinstance(svd_rank, int):
            rank = min(svd_rank, U.shape[1])
        else:
            rank = X.shape[1]

        U = U[:, :rank]
        V = V[:, :rank]
        s = s[:rank]

        return U, s, V

    @staticmethod
    def _build_lowrank_op(U, s, V, Y):
        """
        Private method that computes the lowrank operator from the singular
        value decomposition of matrix X and the matrix Y.

        .. math::

            \\mathbf{\\tilde{A}} =
            \\mathbf{U}^* \\mathbf{Y} \\mathbf{X}^\\dagger \\mathbf{U} =
            \\mathbf{U}^* \\mathbf{Y} \\mathbf{V} \\mathbf{S}^{-1}

        :param numpy.ndarray U: 2D matrix that contains the left-singular
            vectors of X, stored by column.
        :param numpy.ndarray s: 1D array that contains the singular values of X.
        :param numpy.ndarray V: 2D matrix that contains the right-singular
            vectors of X, stored by row.
        :param numpy.ndarray Y: input matrix Y.
        :return: the lowrank operator
        :rtype: numpy.ndarray
        """
        return U.T.conj().dot(Y).dot(V) * np.reciprocal(s)

    @staticmethod
    def _eig_from_lowrank_op(Atilde, Y, U, s, V, exact, rescale_mode=None):
        """
        Private method that computes eigenvalues and eigenvectors of the
        high-dimensional operator from the low-dimensional operator and the
        input matrix.

        :param numpy.ndarray Atilde: the lowrank operator.
        :param numpy.ndarray Y: input matrix Y.
        :param numpy.ndarray U: 2D matrix that contains the left-singular
            vectors of X, stored by column.
        :param numpy.ndarray s: 1D array that contains the singular values of X.
        :param numpy.ndarray V: 2D matrix that contains the right-singular
            vectors of X, stored by row.
        :param bool exact: if True, the exact modes are computed; otherwise,
            the projected ones are computed.
        :param numpy.array rescale_mode: None means no rescaling, 'auto' means
            automatic rescaling using SV, otherwise the user chooses the
            preferred scaling.
        :return: eigenvalues, eigenvectors
        :rtype: numpy.ndarray, numpy.ndarray
        """

        if rescale_mode is None:
            # scaling isn't required
            Ahat = Atilde
        else:
            # rescale using the singular values (as done in the paper)
            if rescale_mode == 'auto':
                scaling_factors_array = s.copy()
            # rescale using custom values
            else:
                scaling_factors_array = rescale_mode

            factors_inv_sqrt = np.diag(np.power(scaling_factors_array, -0.5))
            factors_sqrt = np.diag(np.power(scaling_factors_array, 0.5))
            Ahat = factors_inv_sqrt.dot(atilde).dot(factors_sqrt)

        lowrank_eigenvalues, lowrank_eigenvectors = np.linalg.eig(Ahat)

        # Compute the eigenvectors of the high-dimensional operator
        if exact:
            eigenvectors = (
                (Y.dot(V) * np.reciprocal(s)).dot(lowrank_eigenvectors))
        else:
            eigenvectors = U.dot(lowrank_eigenvectors)

        # The eigenvalues are the same
        eigenvalues = lowrank_eigenvalues

        return eigenvalues, eigenvectors

    def _compute_amplitudes(self, modes, snapshots, eigs, opt):
        """
        Compute the amplitude coefficients. If `opt` is False the amplitudes
        are computed by minimizing the error between the modes and the first
        snapshot; if `opt` is True the amplitudes are computed by minimizing
        the error between the modes and all the snapshots, at the expense of
        bigger computational cost.

        :param numpy.ndarray modes: 2D matrix that contains the modes, stored
            by column.
        :param numpy.ndarray snapshots: 2D matrix that contains the original
            snapshots, stored by column.
        :param numpy.ndarray eigs: array that contains the eigenvalues of the
            linear operator.
        :param bool opt: flag for computing the optimal amplitudes of the DMD
            modes, minimizing the error between the time evolution and all
            the original snapshots. If false the amplitudes are computed
            using only the initial condition, that is snapshots[0].
        :return: the amplitudes array
        :rtype: numpy.ndarray

        References for optimal amplitudes:
        Jovanovic et al. 2014, Sparsity-promoting dynamic mode decomposition,
        https://hal-polytechnique.archives-ouvertes.fr/hal-00995141/document
        """
        if opt:
            # compute the vandermonde matrix
            omega = old_div(np.log(eigs), self.original_time['dt'])
            vander = np.exp(
                np.multiply(*np.meshgrid(omega, self.dmd_timesteps))).T

            # perform svd on all the snapshots
            U, s, V = np.linalg.svd(self._snapshots, full_matrices=False)

            P = np.multiply(
                np.dot(modes.conj().T, modes),
                np.conj(np.dot(vander, vander.conj().T)))
            tmp = (np.dot(np.dot(U, np.diag(s)), V)).conj().T
            q = np.conj(np.diag(np.dot(np.dot(vander, tmp), modes)))

            # b optimal
            a = np.linalg.solve(P, q)
        else:
            a = np.linalg.lstsq(modes, snapshots.T[0], rcond=None)[0]

        return a

    def plot_eigs(self,
                  show_axes=True,
                  show_unit_circle=True,
                  figsize=(8, 8),
                  title=''):
        """
        Plot the eigenvalues.

        :param bool show_axes: if True, the axes will be showed in the plot.
            Default is True.
        :param bool show_unit_circle: if True, the circle with unitary radius
            and center in the origin will be showed. Default is True.
        :param tuple(int,int) figsize: tuple in inches defining the figure
            size. Default is (8, 8).
        :param str title: title of the plot.
        """
        if self._eigs is None:
            raise ValueError('The eigenvalues have not been computed.'
                             'You have to perform the fit method.')

        plt.figure(figsize=figsize)
        plt.title(title)
        plt.gcf()
        ax = plt.gca()

        points, = ax.plot(
            self._eigs.real, self._eigs.imag, 'bo', label='Eigenvalues')

        # set limits for axis
        limit = np.max(np.ceil(np.absolute(self._eigs)))
        ax.set_xlim((-limit, limit))
        ax.set_ylim((-limit, limit))

        plt.ylabel('Imaginary part')
        plt.xlabel('Real part')

        if show_unit_circle:
            unit_circle = plt.Circle(
                (0., 0.),
                1.,
                color='green',
                fill=False,
                label='Unit circle',
                linestyle='--')
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
        if show_unit_circle:
            ax.add_artist(
                plt.legend(
                    [points, unit_circle], ['Eigenvalues', 'Unit circle'],
                    loc=1))
        else:
            ax.add_artist(plt.legend([points], ['Eigenvalues'], loc=1))

        plt.show()

    def plot_modes_2D(self,
                      index_mode=None,
                      filename=None,
                      x=None,
                      y=None,
                      order='C',
                      figsize=(8, 8)):
        """
        Plot the DMD Modes.

        :param index_mode: the index of the modes to plot. By default, all
            the modes are plotted.
        :type index_mode: int or sequence(int)
        :param str filename: if specified, the plot is saved at `filename`.
        :param numpy.ndarray x: domain abscissa.
        :param numpy.ndarray y: domain ordinate
        :param order: read the elements of snapshots using this index order,
            and place the elements into the reshaped array using this index
            order.  It has to be the same used to store the snapshot. 'C' means
            to read/ write the elements using C-like index order, with the last
            axis index changing fastest, back to the first axis index changing
            slowest.  'F' means to read / write the elements using Fortran-like
            index order, with the first index changing fastest, and the last
            index changing slowest.  Note that the 'C' and 'F' options take no
            account of the memory layout of the underlying array, and only
            refer to the order of indexing.  'A' means to read / write the
            elements in Fortran-like index order if a is Fortran contiguous in
            memory, C-like order otherwise.
        :type order: {'C', 'F', 'A'}, default 'C'.
        :param tuple(int,int) figsize: tuple in inches defining the figure
            size. Default is (8, 8).
        """
        if self._modes is None:
            raise ValueError('The modes have not been computed.'
                             'You have to perform the fit method.')

        if x is None and y is None:
            if self._snapshots_shape is None:
                raise ValueError(
                    'No information about the original shape of the snapshots.')

            if len(self._snapshots_shape) != 2:
                raise ValueError(
                    'The dimension of the input snapshots is not 2D.')

        # If domain dimensions have not been passed as argument,
        # use the snapshots dimensions
        if x is None and y is None:
            x = np.arange(self._snapshots_shape[0])
            y = np.arange(self._snapshots_shape[1])

        xgrid, ygrid = np.meshgrid(x, y)

        if index_mode is None:
            index_mode = list(range(self._modes.shape[1]))
        elif isinstance(index_mode, int):
            index_mode = [index_mode]

        if filename:
            basename, ext = splitext(filename)

        for idx in index_mode:
            fig = plt.figure(figsize=figsize)
            fig.suptitle('DMD Mode {}'.format(idx))

            real_ax = fig.add_subplot(1, 2, 1)
            imag_ax = fig.add_subplot(1, 2, 2)

            mode = self._modes.T[idx].reshape(xgrid.shape, order=order)

            real = real_ax.pcolor(
                xgrid,
                ygrid,
                mode.real,
                cmap='jet',
                vmin=mode.real.min(),
                vmax=mode.real.max())
            imag = imag_ax.pcolor(
                xgrid,
                ygrid,
                mode.imag,
                vmin=mode.imag.min(),
                vmax=mode.imag.max())

            fig.colorbar(real, ax=real_ax)
            fig.colorbar(imag, ax=imag_ax)

            real_ax.set_aspect('auto')
            imag_ax.set_aspect('auto')

            real_ax.set_title('Real')
            imag_ax.set_title('Imag')

            # padding between elements
            plt.tight_layout(pad=2.)

            if filename:
                plt.savefig('{0}.{1}{2}'.format(basename, idx, ext))
                plt.close(fig)

        if not filename:
            plt.show()

    def plot_snapshots_2D(self,
                          index_snap=None,
                          filename=None,
                          x=None,
                          y=None,
                          order='C',
                          figsize=(8, 8)):
        """
        Plot the snapshots.

        :param index_snap: the index of the snapshots to plot. By default, all
            the snapshots are plotted.
        :type index_snap: int or sequence(int)
        :param str filename: if specified, the plot is saved at `filename`.
        :param numpy.ndarray x: domain abscissa.
        :param numpy.ndarray y: domain ordinate
        :param order: read the elements of snapshots using this index order,
            and place the elements into the reshaped array using this index
            order.  It has to be the same used to store the snapshot. 'C' means
            to read/ write the elements using C-like index order, with the last
            axis index changing fastest, back to the first axis index changing
            slowest.  'F' means to read / write the elements using Fortran-like
            index order, with the first index changing fastest, and the last
            index changing slowest.  Note that the 'C' and 'F' options take no
            account of the memory layout of the underlying array, and only
            refer to the order of indexing.  'A' means to read / write the
            elements in Fortran-like index order if a is Fortran contiguous in
            memory, C-like order otherwise.
        :type order: {'C', 'F', 'A'}, default 'C'.
        :param tuple(int,int) figsize: tuple in inches defining the figure
            size. Default is (8, 8).
        """
        if self._snapshots is None:
            raise ValueError('Input snapshots not found.')

        if x is None and y is None:
            if self._snapshots_shape is None:
                raise ValueError(
                    'No information about the original shape of the snapshots.')

            if len(self._snapshots_shape) != 2:
                raise ValueError(
                    'The dimension of the input snapshots is not 2D.')

        # If domain dimensions have not been passed as argument,
        # use the snapshots dimensions
        if x is None and y is None:
            x = np.arange(self._snapshots_shape[0])
            y = np.arange(self._snapshots_shape[1])

        xgrid, ygrid = np.meshgrid(x, y)

        if index_snap is None:
            index_snap = list(range(self._snapshots.shape[1]))
        elif isinstance(index_snap, int):
            index_snap = [index_snap]

        if filename:
            basename, ext = splitext(filename)

        for idx in index_snap:
            fig = plt.figure(figsize=figsize)
            fig.suptitle('Snapshot {}'.format(idx))

            snapshot = (self._snapshots.T[idx].real.reshape(
                xgrid.shape, order=order))

            contour = plt.pcolor(
                xgrid,
                ygrid,
                snapshot,
                vmin=snapshot.min(),
                vmax=snapshot.max())

            fig.colorbar(contour)

            if filename:
                plt.savefig('{0}.{1}{2}'.format(basename, idx, ext))
                plt.close(fig)

        if not filename:
            plt.show()
