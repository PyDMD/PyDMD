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

from .dmdoperator import DMDOperator

class DMDBase(object):
    """
    Dynamic Mode Decomposition base class.

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
    :param opt: If True, amplitudes are computed like in optimized DMD  (see
        :func:`~dmdbase.DMDBase._compute_amplitudes` for reference). If
        False, amplitudes are computed following the standard algorithm. If
        `opt` is an integer, it is used as the (temporal) index of the snapshot
        used to compute DMD modes amplitudes (following the standard algorithm).
        The reconstruction will generally be better in time instants near the
        chosen snapshot; however increasing `opt` may lead to wrong results when
        the system presents small eigenvalues. For this reason a manual
        selection of the number of eigenvalues considered for the analyisis may
        be needed (check `svd_rank`). Also setting `svd_rank` to a value between
        0 and 1 may give better results. Default is False.
    :type opt: bool or int
    :param rescale_mode: Scale Atilde as shown in
            10.1016/j.jneumeth.2015.10.010 (section 2.4) before computing its
            eigendecomposition. None means no rescaling, 'auto' means automatic
            rescaling using singular values, otherwise the scaling factors.
    :type rescale_mode: {'auto'} or None or numpy.ndarray
    :param bool forward_backward: If True, the low-rank operator is computed
        like in fbDMD (reference: https://arxiv.org/abs/1507.02264). Default is
        False.

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
        rescale_mode=None, forward_backward=False):
        self._Atilde = DMDOperator(svd_rank=svd_rank, exact=exact,
            rescale_mode=rescale_mode, forward_backward=forward_backward)

        self._tlsq_rank = tlsq_rank
        self.original_time = None
        self.dmd_time = None
        self._opt = opt

        self._b = None  # amplitudes
        self._snapshots = None
        self._snapshots_shape = None

    @property
    def opt(self):
        return self._opt

    @property
    def tlsq_rank(self):
        return self._tlsq_rank

    @property
    def svd_rank(self):
        return self.operator._svd_rank

    @property
    def rescale_mode(self):
        return self.operator._rescale_mode

    @property
    def exact(self):
        return self.operator._exact

    @property
    def forward_backward(self):
        return self.operator._forward_backward

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
        return self.operator.modes

    @property
    def atilde(self):
        """
        Get the reduced Koopman operator A, called A tilde.

        :return: the reduced Koopman operator A.
        :rtype: numpy.ndarray
        """
        return self.operator.as_numpy_array

    @property
    def operator(self):
        """
        Get the instance of DMDOperator.

        :return: the instance of DMDOperator
        :rtype: DMDOperator
        """
        return self._Atilde

    @property
    def eigs(self):
        """
        Get the eigenvalues of A tilde.

        :return: the eigenvalues from the eigendecomposition of `atilde`.
        :rtype: numpy.ndarray
        """
        return self.operator.eigenvalues

    def _translate_eigs_exponent(self, tpow):
        """
        Get the eigenvalues of A tilde.

        :param tpow: the exponent(s) of Sigma in the original DMD formula.
        :type tpow: int or np.ndarray
        :return: the eigenvalues from the eigendecomposition of `atilde`.
        :rtype: numpy.ndarray
        """

        if isinstance(self.opt, bool):
            amplitudes_snapshot_index = 0
        else:
            amplitudes_snapshot_index = self.opt

        if amplitudes_snapshot_index < 0:
            # we take care of negative indexes: -n becomes T - n
            return tpow - (self.snapshots.shape[1] + amplitudes_snapshot_index)
        else:
            return tpow - amplitudes_snapshot_index

    @property
    def dynamics(self):
        """
        Get the time evolution of each mode.

        .. math::

            \\mathbf{x}(t) \\approx
            \\sum_{k=1}^{r} \\boldsymbol{\\phi}_{k} \\exp \\left( \\omega_{k} t \\right) b_{k} =
            \\sum_{k=1}^{r} \\boldsymbol{\\phi}_{k} \\left( \\lambda_{k} \\right)^{\\left( t / \\Delta t \\right)} b_{k}

        :return: the matrix that contains all the time evolution, stored by
            row.
        :rtype: numpy.ndarray

        """
        temp = np.outer(self.eigs, np.ones(self.dmd_timesteps.shape[0]))
        tpow = old_div(self.dmd_timesteps - self.original_time['t0'],
                       self.original_time['dt'])

        # The new formula is x_(k+j) = \Phi \Lambda^k \Phi^(-1) x_j.
        # Since j is fixed, for a given snapshot "u" we have the following
        # formula:
        # x_u = \Phi \Lambda^{u-j} \Phi^(-1) x_j
        # Therefore tpow must be scaled appropriately.
        tpow = self._translate_eigs_exponent(tpow)

        return (np.power(temp, tpow) * self._b[:, None])

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
            warnings.warn(
                "Input data matrix X has condition number {}. "
                "Consider preprocessing data, passing in augmented data matrix, or regularization methods."
                .format(cond_number))

        return snapshots, snapshots_shape

    def _compute_amplitudes(self):
        """
        Compute the amplitude coefficients. If `self.opt` is False the
        amplitudes are computed by minimizing the error between the modes and
        the first snapshot; if `self.opt` is True the amplitudes are computed by
        minimizing the error between the modes and all the snapshots, at the
        expense of bigger computational cost.

        This method uses the class variables self._snapshots (for the
        snapshots), self.modes and self.eigs.

        :return: the amplitudes array
        :rtype: numpy.ndarray

        References for optimal amplitudes:
        Jovanovic et al. 2014, Sparsity-promoting dynamic mode decomposition,
        https://hal-polytechnique.archives-ouvertes.fr/hal-00995141/document
        """
        if isinstance(self.opt, bool) and self.opt:
            # compute the vandermonde matrix
            omega = old_div(np.log(self.eigs), self.original_time['dt'])
            vander = np.exp(
                np.multiply(*np.meshgrid(omega, self.dmd_timesteps))).T

            # perform svd on all the snapshots
            U, s, V = np.linalg.svd(self._snapshots, full_matrices=False)

            P = np.multiply(np.dot(self.modes.conj().T, self.modes),
                            np.conj(np.dot(vander,
                                           vander.conj().T)))

            tmp = np.linalg.multi_dot([U, np.diag(s), V]).conj().T
            q = np.conj(np.diag(np.linalg.multi_dot([vander, tmp, self.modes])))

            # b optimal
            a = np.linalg.solve(P, q)
        else:
            if isinstance(self.opt, bool):
                amplitudes_snapshot_index = 0
            else:
                amplitudes_snapshot_index = self.opt

            a = np.linalg.lstsq(self.modes,
                self._snapshots.T[amplitudes_snapshot_index],
                rcond=None)[0]

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
        if self.eigs is None:
            raise ValueError('The eigenvalues have not been computed.'
                             'You have to perform the fit method.')

        plt.figure(figsize=figsize)
        plt.title(title)
        plt.gcf()
        ax = plt.gca()

        points, = ax.plot(self.eigs.real,
                          self.eigs.imag,
                          'bo',
                          label='Eigenvalues')

        # set limits for axis
        limit = np.max(np.ceil(np.absolute(self.eigs)))
        ax.set_xlim((-limit, limit))
        ax.set_ylim((-limit, limit))

        plt.ylabel('Imaginary part')
        plt.xlabel('Real part')

        if show_unit_circle:
            unit_circle = plt.Circle((0., 0.),
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
            ax.annotate('',
                        xy=(np.max([limit * 0.8, 1.]), 0.),
                        xytext=(np.min([-limit * 0.8, -1.]), 0.),
                        arrowprops=dict(arrowstyle="->"))
            ax.annotate('',
                        xy=(0., np.max([limit * 0.8, 1.])),
                        xytext=(0., np.min([-limit * 0.8, -1.])),
                        arrowprops=dict(arrowstyle="->"))

        # legend
        if show_unit_circle:
            ax.add_artist(
                plt.legend([points, unit_circle],
                           ['Eigenvalues', 'Unit circle'],
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
        if self.modes is None:
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
            index_mode = list(range(self.modes.shape[1]))
        elif isinstance(index_mode, int):
            index_mode = [index_mode]

        if filename:
            basename, ext = splitext(filename)

        for idx in index_mode:
            fig = plt.figure(figsize=figsize)
            fig.suptitle('DMD Mode {}'.format(idx))

            real_ax = fig.add_subplot(1, 2, 1)
            imag_ax = fig.add_subplot(1, 2, 2)

            mode = self.modes.T[idx].reshape(xgrid.shape, order=order)

            real = real_ax.pcolor(xgrid,
                                  ygrid,
                                  mode.real,
                                  cmap='jet',
                                  vmin=mode.real.min(),
                                  vmax=mode.real.max())
            imag = imag_ax.pcolor(xgrid,
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

            snapshot = (self._snapshots.T[idx].real.reshape(xgrid.shape,
                                                            order=order))

            contour = plt.pcolor(xgrid,
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
