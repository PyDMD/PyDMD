"""
Derived module from dmdbase.py for multi-resolution dmd.

Reference:
- Kutz, J. Nathan, Xing Fu, and Steven L. Brunton. Multiresolution Dynamic Mode
Decomposition. SIAM Journal on Applied Dynamical Systems 15.2 (2016): 713-735.
"""
from copy import deepcopy
from functools import partial

import numpy as np
from past.utils import old_div

from .dmd_modes_tuner import select_modes
from .dmdbase import DMDBase
from .linalg import build_linalg_module, cast_as_array
from .snapshots import Snapshots


class BinaryTree:
    """Simple Binary tree"""

    def __init__(self, depth):
        self.depth = depth
        self.tree = [None] * len(self)

    def __len__(self):
        return 2 ** (self.depth + 1) - 1

    def __getitem__(self, val):
        level_, bin_ = val
        if level_ > self.depth:
            raise ValueError(
                """The level input parameter ({}) has to be less or equal than
                the max_level ({}). Remember that the starting
                index is 0""".format(
                    level_, self.depth
                )
            )

        if bin_ >= 2**level_:
            raise ValueError("Invalid node")

        return self.tree[2**level_ + bin_ - 1]

    def __setitem__(self, val, item):
        level_, bin_ = val
        self.tree[2**level_ + bin_ - 1] = item

    def __iter__(self):
        return self.tree.__iter__()

    @property
    def levels(self):
        return tuple(range(self.depth + 1))

    def index_leaves(self, level):
        return tuple(range(0, 2**level))


class MrDMD(DMDBase):
    """
    Multi-resolution Dynamic Mode Decomposition

    :param dmd: DMD instance(s) used to analyze the snapshots provided. See also
        the documentation for :meth:`_dmd_builder`.
    :type dmd: DMDBase or list or tuple or function
    :param int max_cycles: the maximum number of mode oscillations in any given
        time scale. Default is 1.
    :param int max_level: the maximum level (inclusive). For instance,
        `max_level=4` means that we are going to have levels `0`, `1`, `2`, `3`
        and `4`. Default is 2.
    """

    def __init__(self, dmd, max_level=2, max_cycles=1):
        self.dmd = dmd
        self.max_cycles = max_cycles
        self.max_level = max_level
        self._build_tree()

    def __iter__(self):
        return self.dmd_tree.__iter__()

    @property
    def modes(self):
        """
        Get the matrix containing the DMD modes, stored by column.

        :return: the matrix containing the DMD modes.
        :rtype: numpy.ndarray
        """
        linalg_module = build_linalg_module(self.snapshots)
        return linalg_module.cat(
            tuple(self.partial_modes(i) for i in range(self.max_level + 1)),
            axis=-1,
        )

    @property
    def dynamics(self):
        """
        Get the time evolution of each mode.

        :return: the matrix that contains all the time evolution, stored by
                row.
        :rtype: numpy.ndarray
        """
        linalg_module = build_linalg_module(self.snapshots)
        return linalg_module.cat(
            tuple(self.partial_dynamics(i) for i in range(self.max_level + 1)),
            axis=-2,
        )

    @property
    def eigs(self):
        """
        Get the eigenvalues of A tilde.

        :return: the eigenvalues from the eigendecomposition of `atilde`.
        :rtype: numpy.ndarray
        """
        linalg_module = build_linalg_module(self.snapshots)
        return linalg_module.cat(tuple(dmd.eigs for dmd in self), axis=-1)

    @property
    def modes_activation_bitmask(self):
        raise RuntimeError("This feature has not been implemented yet.")

    @modes_activation_bitmask.setter
    def modes_activation_bitmask(self, value):
        raise RuntimeError("This feature has not been implemented yet.")

    @property
    def reconstructed_data(self):
        """
        Get the reconstructed data.

        :return: the matrix that contains the reconstructed snapshots.
        :rtype: numpy.ndarray
        """
        linalg_module = build_linalg_module(self.snapshots)
        return cast_as_array(
            [
                linalg_module.cat(
                    tuple(
                        self.dmd_tree[level, leaf].reconstructed_data
                        for leaf in self.dmd_tree.index_leaves(level)
                    ),
                    -1,
                )
                for level in self.dmd_tree.levels
            ]
        ).sum(0)

    def _dmd_builder(self):
        """
        Builds a function which takes in input a level and a leaf count
        (i.e. coordinates inside the binary tree) and produces an appropriate
        DMD instance according to the criteria specified in `self.dmd`.

        Criteria supported:

        - A function which takes two parameters `level` and `leaf`;
        - List/tuple of DMD instances (length must be equal to `max_level+1`);
        - A DMD instance (which is used for all the levels and leaves).

        Example 0 (one DMD):

        .. code-block:: python

            >>> # this SpDMD is used for all the levels, for all the leaves
            >>> MrDMD(dmd=SpDMD(), max_level=5).fit(X)

        Example 1 (simple function which adapts the parameter d of HankelDMD
        to the current level of the tree):

        .. code-block:: python

            >>> def build_dmds(level, leaf):
            ...     d = 30 - 2*level
            ...     return HankelDMD(d=d)
            >>> MrDMD(dmd=build_dmds, max_level=5).fit(X)

        Example 2 (we use a different kind of DMD if we are near the middle part
        of the time window):

        .. code-block:: python

            >>> # you can name the function however you prefer
            >>> def my_dmds(level, leaf):
            ...     level_size = pow(2,level)
            ...     distance_from_middle = abs(leaf - level_size // 2)
            ...     # we choose 2 as a random threshold
            ...     if distance_from_middle < 2:
            ...         return HankelDMD(d=5)
            ...     else:
            ...         return DMD(svd_rank=3)
            >>> MrDMD(dmd=my_dmds, max_level=5).fit(X)

        Example 3 (tuple of DMDs):

        .. code-block:: python

            >>> dmds_list = [DMD(svd_rank=10) for i in range(6) if i < 3
                                else DMD(svd_rank=2)]
            >>> MrDMD(dmd=dmds_list, max_level=5).fit(X)

        :return: A function which can be used to spawn DMD instances according
            to the level and leaf.
        :rtype: func
        """
        if callable(self.dmd):
            builder_func = self.dmd
        elif isinstance(self.dmd, (list, tuple)):
            if len(self.dmd) != self.max_level + 1:
                raise ValueError(
                    """
Expected one item per level, got {} out of {} levels.""".format(
                        len(self.dmd), self.max_level
                    )
                )

            def builder_func(level, *args):
                return deepcopy(self.dmd[level])

        elif isinstance(self.dmd, DMDBase):

            def builder_func(*args):
                return deepcopy(self.dmd)

        return builder_func

    def _build_tree(self):
        """
        Build the internal binary tree that contain the DMD subclasses.
        """
        self.dmd_tree = BinaryTree(self.max_level)

        # we build a function which takes the level and leaf and returns a DMD
        builder_func = self._dmd_builder()

        # Empty init
        for level in self.dmd_tree.levels:
            for leaf in self.dmd_tree.index_leaves(level):
                # we construct the dmd before in order to trigger a better
                # exception in case the function fails (otherwise it would
                # be an IndexError)
                self.dmd_tree[level, leaf] = builder_func(level, leaf)

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
        for level in self.dmd_tree.levels:
            for leaf in self.dmd_tree.index_leaves(level):
                local_times = self.partial_time_interval(level, leaf)
                if (
                    local_times["t0"] <= t0 < local_times["tend"]
                    or local_times["t0"] < tend <= local_times["tend"]
                    or (t0 <= local_times["t0"] and tend >= local_times["tend"])
                ):
                    indexes.append((level, leaf))

        indexes = np.unique(indexes, axis=0)
        return indexes

    def time_window_eigs(self, t0, tend):
        """
        Get the eigenvalues relative to the modes of the bins embedded
        (partially or totally) in a given time window.

        :param float t0: start time of the window.
        :param float tend: end time of the window.
        :return: the eigenvalues for that time window.
        :rtype: numpy.ndarray
        """
        indexes = self.time_window_bins(t0, tend)
        linalg_module = build_linalg_module(self.snapshots)
        return linalg_module.cat(
            tuple(self.dmd_tree[idx].eigs for idx in indexes), -1
        )

    def time_window_frequency(self, t0, tend):
        """
        Get the frequencies relative to the modes of the bins embedded
        (partially or totally) in a given time window.

        :param float t0: start time of the window.
        :param float tend: end time of the window.
        :return: the frequencies for that time window.
        :rtype: numpy.ndarray
        """
        indexes = self.time_window_bins(t0, tend)
        linalg_module = build_linalg_module(self.snapshots)
        return linalg_module.cat(
            tuple(self.dmd_tree[idx].frequency for idx in indexes), -1
        )

    def time_window_growth_rate(self, t0, tend):
        """
        Get the growth rate values relative to the modes of the bins embedded
        (partially or totally) in a given time window.

        :param float t0: start time of the window.
        :param float tend: end time of the window.
        :return: the Floquet values for that time window.
        :rtype: numpy.ndarray
        """
        indexes = self.time_window_bins(t0, tend)
        linalg_module = build_linalg_module(self.snapshots)
        return linalg_module.cat(
            tuple(self.dmd_tree[idx].growth_rate for idx in indexes), -1
        )

    def time_window_amplitudes(self, t0, tend):
        """
        Get the amplitudes relative to the modes of the bins embedded
        (partially or totally) in a given time window.

        :param float t0: start time of the window.
        :param float tend: end time of the window.
        :return: the amplitude of the modes for that time window.
        :rtype: numpy.ndarray
        """
        indexes = self.time_window_bins(t0, tend)
        linalg_module = build_linalg_module(self.snapshots)
        return linalg_module.cat(
            tuple(self.dmd_tree[idx].amplitudes for idx in indexes), -1
        )

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

        :return: the selected modes stored by columns
        :rtype: numpy.ndarray
        """
        leaves = self.dmd_tree.index_leaves(level) if node is None else [node]
        linalg_module = build_linalg_module(self.snapshots)
        return linalg_module.cat(
            tuple(self.dmd_tree[level, leaf].modes for leaf in leaves), -1
        )

    def partial_dynamics(self, level, node=None):
        """
        Return the time evolution of the specific `level` and of the specific
        `node`; if `node` is not specified, the method returns the time
        evolution of the given `level` (all the nodes). The dynamics are always
        reported to the original time window.

        :param int level: the index of the level from where the time evolution
            is extracted.
        :param int node: the index of the node from where the time evolution is
            extracted; if None, the time evolution is extracted from all the
            nodes of the given level. Default is None.

        :return: the selected dynamics stored by row
        :rtype: numpy.ndarray
        """
        leaves = self.dmd_tree.index_leaves(level) if node is None else [node]
        linalg_module = build_linalg_module(self.snapshots)
        return linalg_module.block_diag(
            *tuple(
                dmd.dynamics
                for dmd in map(lambda leaf: self.dmd_tree[level, leaf], leaves)
            )
        )

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

        :return: the selected eigs
        :rtype: numpy.ndarray
        """
        leaves = self.dmd_tree.index_leaves(level) if node is None else [node]
        linalg_module = build_linalg_module(self.snapshots)
        return linalg_module.cat(
            tuple(self.dmd_tree[level, leaf].eigs for leaf in leaves), -1
        )

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

        :return: the selected reconstruction from dmd operators
        :rtype: numpy.ndarray
        """
        modes = self.partial_modes(level, node)
        dynamics = self.partial_dynamics(level, node)

        linalg_module = build_linalg_module(self.snapshots)
        return linalg_module.dot(modes, dynamics)

    def partial_time_interval(self, level, leaf):
        """
        Evaluate the start and end time and the period of a given bin.

        :param int level: the level in the binary tree.
        :param int node: the node id.
        :return: the start and end time and the period of the bin
        :rtype: dictionary
        """
        if level > self.max_level:
            raise ValueError(
                "The level input parameter ({}) has to be less than the "
                "max_level ({}). Remember that the starting index is 0".format(
                    level, self.max_level
                )
            )

        if leaf >= 2**level:
            raise ValueError("Invalid node")

        full_period = self.original_time["tend"] - self.original_time["t0"]
        period = full_period / 2**level
        t0 = self.original_time["t0"] + period * leaf
        tend = t0 + period
        return {"t0": t0, "tend": tend, "delta": period}

    def enumerate(self):
        """

        Example:

        >>> mrdmd = MrDMD(DMD())
        >>> mrdmd.fit(X)
        >>> for level, leaf, dmd in mrdmd:
        >>>     print(level, leaf, dmd.eigs)

        """
        for level in self.dmd_tree.levels:
            for leaf in self.dmd_tree.index_leaves(level):
                yield level, leaf, self.dmd_tree[level, leaf]

    def fit(self, X, batch=False):
        """
        Compute the Dynamic Modes Decomposition to the input data.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        """
        self._reset()

        self._snapshots_holder = Snapshots(X, batch=batch)
        linalg_module = build_linalg_module(self.snapshots)

        # Redefine max level if it is too big.
        lvl_threshold = (
            int(np.log(self.snapshots.shape[-1] / 4.0) / np.log(2.0)) + 1
        )
        if self.max_level > lvl_threshold:
            self.max_level = lvl_threshold
            self._build_tree()
            print(
                "Too many levels... "
                "Redefining `max_level` to {}".format(self.max_level)
            )

        def slow_modes(dmd, rho):
            return linalg_module.abs(linalg_module.log(dmd.eigs)) < rho * 2 * np.pi

        X = linalg_module.new_array(self.snapshots)
        for level in self.dmd_tree.levels:
            n_leaf = 2**level
            Xs = linalg_module.array_split(X, n_leaf, axis=-1)

            for leaf, x in enumerate(Xs):
                current_dmd = self.dmd_tree[level, leaf]
                current_dmd.fit(x, batch=batch)

                rho = old_div(float(self.max_cycles), x.shape[-1])
                slow_modes_selector = partial(slow_modes, rho=rho)

                select_modes(current_dmd, slow_modes_selector)

            newX = linalg_module.cat(
                tuple(
                    self.dmd_tree[level, leaf].reconstructed_data
                    for leaf in self.dmd_tree.index_leaves(level)
                ), axis=-1
            )
            X = X - linalg_module.to(X, newX)

        self._set_initial_time_dictionary(
            dict(t0=0, tend=self.snapshots.shape[-1], dt=1)
        )

        return self
