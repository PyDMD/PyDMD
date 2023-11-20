"""
Derived module from dmdbase.py for Optimized DMD and Bagging, Optimized DMD
(BOP-DMD).

References:
- Travis Askham and J. Nathan Kutz. Variable projection methods for an
optimized dynamic mode decomposition. SIAM Journal on Applied Dynamical
Systems, 17, 2018.
- Diya Sashidhar and J. Nathan Kutz. Bagging, optimized dynamic mode
decomposition (bop-dmd) for robust, stable forecasting with spatial and
temporal uncertainty-quantification. 2021. arXiv:2107.10878.
"""

import warnings
from collections import OrderedDict
from inspect import isfunction

import numpy as np
from scipy.linalg import qr
from scipy.sparse import csr_matrix

from .dmdbase import DMDBase
from .dmdoperator import DMDOperator
from .snapshots import Snapshots
from .utils import compute_rank, compute_svd


class BOPDMDOperator(DMDOperator):
    """
    BOP-DMD operator.

    :param compute_A: Flag that determines whether or not to compute the full
        Koopman operator A.
    :type compute_A: bool
    :param use_proj: Flag that determines the type of computation to perform.
        If True, fit input data projected onto the first svd_rank POD modes or
        columns of proj_basis if provided. If False, fit the full input data.
    :type use_proj: bool
    :param init_alpha: Initial guess for the continuous-time DMD eigenvalues.
    :type init_alpha: numpy.ndarray
    :param proj_basis: Orthogonal basis for projection, where each column of
        proj_basis contains a basis mode.
    :type proj_basis: numpy.ndarray
    :param num_trials: Number of BOP-DMD trials to perform. If num_trials is a
        positive integer, num_trials BOP-DMD trials are performed. Otherwise,
        standard optimized dmd is performed.
    :type num_trials: int
    :param trial_size: Size of the randomly selected subset of observations to
        use for each trial of bagged optimized dmd (BOP-DMD). If trial_size is
        a positive integer, trial_size many observations will be used per
        trial. If trial_size is a float between 0 and 1, int(trial_size * m)
        many observations will be used per trial, where m denotes the total
        number of data points observed. Note that any other type of input for
        trial_size will yield an error.
    :type trial_size: int or float
    :param eig_sort: Method used to sort eigenvalues (and modes accordingly)
        when performing BOP-DMD. Eigenvalues will be sorted by real part and
        then by imaginary part to break ties if `eig_sort="real"`, by imaginary
        part and then by real part to break ties if `eig_sort="imag"`, or by
        magnitude if `eig_sort="abs"`. If `eig_sort="auto"`, one of the
        previously-mentioned sorting methods is chosen depending on eigenvalue
        variance.
    :type eig_sort: {"real", "imag", "abs", "auto"}
    :param eig_constraints: Set containing desired DMD operator eigenvalue
        constraints. Currently available constraints are `"stable"`, which
        constrains eigenvalues to the left half of the complex plane,
        `"imag"`, which constrains eigenvalues to the imaginary axis, and
        `"conjugate_pairs"`, which enforces that eigenvalues are always
        present with their complex conjugate. Note that constraints may be
        combined if valid. May alternatively be a custom eigenvalue constraint
        function that will be applied to the computed eigenvalues at each step
        of the variable projection routine.
    :type eig_constraints: set(str) or function
    :param bag_warning: Number of consecutive non-converged trials of BOP-DMD
        at which to produce a warning message for the user. Default is 100.
        Use arguments less than or equal to zero for no warning condition.
    :type bag_warning: int
    :param bag_maxfail: Number of consecutive non-converged trials of BOP-DMD
        at which to terminate the fit. Default is -1, no stopping condition.
    :type bag_maxfail: int
    :param init_lambda: Initial value used for the regularization parameter in
        the Levenberg method. Default is 1.0.
        Note: Larger lambda values make the method more like gradient descent.
    :type init_lambda: float
    :param maxlam: Maximum number of of steps used in the inner Levenberg loop,
        i.e. the number of times you increase lambda before quitting. Default
        is 52.
    :type maxlam: int
    :param lamup: The factor by which you increase lambda when searching for an
        appropriate step. Default is 2.0.
    :type lamup: float
    :param use_levmarq: Flag that determines whether you use the Levenberg
        algorithm or the Levenberg-Marquardt algorithm. Default is True,
        use Levenberg-Marquardt.
    :type use_levmarq: bool
    :param maxiter: The maximum number of outer loop iterations to use before
        quitting. Default is 30.
    :type maxiter: int
    :param tol: The tolerance for the relative error in the residual.
        i.e. the program will terminate if
            norm(y-Phi(alpha)*b,'fro')/norm(y,'fro') < tol
        is achieved. Default is 1e-6.
    :type tol: float
    :param eps_stall: The tolerance for detecting a stall.
        i.e. if
            error(iter-1)-error(iter) < eps_stall*err(iter-1)
        the program halts. Default is 1e-12.
    :type eps_stall: float
    :param use_fulljac: Flag that determines whether or not to use the full
        expression for the Jacobian or Kaufman's approximation. Default is
        True, use full expression.
    :type use_fulljac: bool
    :param verbose: Flag that determines whether or not to print warning
        messages that arise during the variable projection routine, and whether
        or not to print information regarding the method's iterative progress.
        Default is False, don't print information.
    :type verbose: bool
    """

    def __init__(
        self,
        compute_A,
        use_proj,
        init_alpha,
        proj_basis,
        num_trials,
        trial_size,
        eig_sort,
        eig_constraints,
        bag_warning,
        bag_maxfail,
        init_lambda=1.0,
        maxlam=52,
        lamup=2.0,
        use_levmarq=True,
        maxiter=30,
        tol=1e-6,
        eps_stall=1e-12,
        use_fulljac=True,
        verbose=False,
    ):
        self._compute_A = compute_A
        self._use_proj = use_proj
        self._init_alpha = init_alpha
        self._proj_basis = proj_basis
        self._num_trials = num_trials
        self._trial_size = trial_size
        self._eig_sort = eig_sort
        self._eig_constraints = eig_constraints
        self._bag_warning = bag_warning
        self._bag_maxfail = bag_maxfail
        self._varpro_opts = [
            init_lambda,
            maxlam,
            lamup,
            use_levmarq,
            maxiter,
            tol,
            eps_stall,
            use_fulljac,
            verbose,
        ]
        self._varpro_opts_warn()

        self._modes = None
        self._eigenvalues = None
        self._modes_std = None
        self._eigenvalues_std = None
        self._amplitudes_std = None
        self._Atilde = None
        self._A = None

    @property
    def varpro_opts(self):
        """
        Get the variable projection options.

        :return: the variable projection options.
        :rtype: tuple
        """
        return self._varpro_opts

    @property
    def A(self):
        """
        Get the full Koopman operator A.

        :return: the full Koopman operator A.
        :rtype: numpy.ndarray
        """
        if not self._compute_A:
            msg = (
                "A not computed during fit. "
                "Set parameter compute_A = True to compute A."
            )
            raise ValueError(msg)
        if self._A is None:
            raise ValueError("You need to call fit before")
        return self._A

    @property
    def amplitudes_std(self):
        """
        Get the amplitudes standard deviation.

        :return: amplitudes standard deviation.
        :rtype: numpy.ndarray
        """
        return self._amplitudes_std

    @property
    def eigenvalues_std(self):
        """
        Get the eigenvalues standard deviation.

        :return: eigenvalues standard deviation.
        :rtype: numpy.ndarray
        """
        return self._eigenvalues_std

    @property
    def modes_std(self):
        """
        Get the modes standard deviation.

        :return: modes standard deviation.
        :rtype: numpy.ndarray
        """
        return self._modes_std

    def _varpro_opts_warn(self):
        """
        Checks the validity of the parameter values in _varpro_opts.
        Throws an error if any parameter value has an invalid type and
        generates a warning if any value lies outside of the recommended range.
        """
        # Generate dictionary of recommended value range for each parameter.
        rec_ranges = OrderedDict()
        rec_ranges["init_lambda"] = [0.0, 1e16]
        rec_ranges["maxlam"] = [0, 200]
        rec_ranges["lamup"] = [1.0, 1e16]
        rec_ranges["use_levmarq"] = [-np.inf, np.inf]
        rec_ranges["maxiter"] = [0, 1e12]
        rec_ranges["tol"] = [0.0, 1e16]
        rec_ranges["eps_stall"] = [-np.inf, 1.0]
        rec_ranges["use_fulljac"] = [-np.inf, np.inf]
        rec_ranges["verbose"] = [-np.inf, np.inf]

        for opt_value, (opt_name, (opt_min, opt_max)) in zip(
            self._varpro_opts, rec_ranges.items()
        ):
            if not isinstance(opt_value, (int, float, bool)):
                raise TypeError("Invalid variable projection option given.")

            if opt_value < opt_min:
                msg = (
                    "Option {} with value {} is less than {}, "
                    "which is not recommended."
                )
                warnings.warn(msg.format(opt_name, opt_value, opt_min))
            elif opt_value > opt_max:
                msg = (
                    "Option {} with value {} is greater than {}, "
                    "which is not recommended."
                )
                warnings.warn(msg.format(opt_name, opt_value, opt_max))

    def _push_eigenvalues(self, eigenvalues):
        """
        Helper function that constrains the given eigenvalues according to
        the arguments found in `self._eig_constraints`. If no constraints were
        given, this function simply returns the given eigenvalues. Applies the
        provided `eig_constraints` function if a function was provided instead
        of a set of constraints.

        :param eigenvalues: Vector of original eigenvalues.
        :type eigenvalues: numpy.ndarray
        :return: Vector of constrained eigenvalues.
        :rtype: numpy.ndarray
        """
        if isfunction(self._eig_constraints):
            return self._eig_constraints(eigenvalues)

        if "conjugate_pairs" in self._eig_constraints:
            num_eigs = len(eigenvalues)
            new_eigs = np.empty(num_eigs, dtype="complex")
            # If given an odd number of eigenvalues, find the eigenvalue with
            # the smallest imaginary part and take it to be a real eigenvalue.
            if num_eigs % 2 == 1:
                ind_single = np.argmin(np.abs(eigenvalues.imag))
                new_eigs[ind_single] = eigenvalues[ind_single].real
                eig_pair_inds = np.delete(np.arange(num_eigs), ind_single)[::2]
            else:
                eig_pair_inds = np.arange(0, num_eigs, 2)
            # Note: a consequence of the OptDMD variable projection process is
            # that complex conjugate pairs naturally appear together in the
            # computed eigenvalue vectors. We thus take advantage of this...
            for i in eig_pair_inds:
                eig_pair = eigenvalues[i : i + 2]
                real_comp = np.mean(eig_pair.real)
                imag_comp = np.mean(np.abs(eig_pair.imag))
                new_eigs[i] = real_comp + 1j * imag_comp
                new_eigs[i + 1] = real_comp - 1j * imag_comp

            eigenvalues = np.copy(new_eigs)

        if "stable" in self._eig_constraints:
            right_half = eigenvalues.real > 0.0
            eigenvalues[right_half] = 1j * eigenvalues[right_half].imag
        elif "imag" in self._eig_constraints:
            eigenvalues = 1j * eigenvalues.imag

        return eigenvalues

    def _exp_function(self, alpha, t):
        """
        Matrix of exponentials.

        :param alpha: Vector of time scalings in the exponent.
        :type alpha: numpy.ndarray
        :param t: Vector of time values.
        :type t: numpy.ndarray
        :return: Matrix A such that A[i, j] = exp(t_i * alpha_j).
        :rtype: numpy.ndarray
        """
        return np.exp(np.outer(t, alpha))

    def _exp_function_deriv(self, alpha, t, i):
        """
        Derivatives of the matrix of exponentials.

        :param alpha: Vector of time scalings in the exponent.
        :type alpha: numpy.ndarray
        :param t: Vector of time values.
        :type t: numpy.ndarray
        :param i: Index in alpha of the derivative variable.
        :type i: int
        :return: Derivatives of Phi(alpha, t) with respect to alpha[i].
        :rtype: scipy.sparse.csr_matrix
        """
        m = len(t)
        n = len(alpha)
        if i < 0 or i > n - 1:
            raise ValueError("Invalid index i given to exp_function_deriv.")
        A = np.multiply(t, np.exp(alpha[i] * t))
        return csr_matrix(
            (A, (np.arange(m), np.full(m, fill_value=i))), shape=(m, n)
        )

    @staticmethod
    def _compute_irank_svd(X, tolrank):
        """
        Helper function that computes and returns the SVD of X with a rank
        truncation of irank, which denotes the number of singular values of
        X greater than tolrank * s1, where s1 is the largest singular value
        of the matrix X.

        :param X: Matrix to decompose.
        :type X: numpy.ndarray
        :param tolrank: Determines the rank of the returned SVD.
        :type tolrank: float
        :return: irank truncated SVD of X.
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
        """
        U, s, Vh = np.linalg.svd(X, full_matrices=False)
        irank = np.sum(s > tolrank * s[0])
        U = U[:, :irank]
        S = np.diag(s[:irank])
        Vh = Vh[:irank]
        return U, S, Vh

    def _argsort_eigenvalues(self, eigs):
        """
        Helper function that computes and returns the indices that sort the
        given array of eigenvalues according to the operator's `eig_sort`
        attribute. Sets `eig_sort` according to eigs if not already done so.

        :param eigs: array of eigenvalues to sort
        :type eigs: numpy.ndarray
        :return: array of indices that sort the given eigenvalues
        :rtype: numpy.ndarray
        """
        # Set the sorting style if eig_sort is "auto".
        # Note: the initial call to this function will set the eig_sort
        # attribute that will be used for the remainder of all fits!
        if self._eig_sort == "auto":
            real_var = np.var(eigs.real)
            imag_var = np.var(eigs.imag)
            abs_var = np.var(np.abs(eigs))
            all_var = [real_var, imag_var, abs_var]
            self._eig_sort = ("real", "imag", "abs")[np.argmax(all_var)]

        # Sort the results according to eig_sort.
        if self._eig_sort == "real":
            return np.argsort(eigs)
        if self._eig_sort == "imag":
            eigs_real_imag_swapped = eigs.imag + (1j * eigs.real)
            return np.argsort(eigs_real_imag_swapped)
        if self._eig_sort == "abs":
            return np.argsort(np.abs(eigs))

        raise ValueError("Provided eig_sort method is not supported.")

    def _bag(self, H, trial_size):
        """
        Given a 2D array of data X, where each row contains a data snapshot,
        randomly sub-selects and returns data snapshots while preserving the
        original snapshot order. Note that if trial_size is a positive integer,
        trial_size many observations will be used per trial. If trial_size is a
        float between 0 and 1, int(trial_size * m) many observations will be
        used per trial, where m denotes the total number of snapshots in X.
        The indices of the sub-selected snapshots are also returned.

        :param H: Full data matrix to be sub-selected from.
        :type H: numpy.ndarray
        :param trial_size: Size of the sub-selection from H.
        :type trial_size: int or float
        :return: Matrix of sub-selected data snapshots, stored in each row,
            and a vector of each snapshots's row index location in H.
        :rtype: numpy.ndarray, numpy.ndarray
        """
        # Ensure that H is a 2D numpy.ndarray.
        if not isinstance(H, np.ndarray) or H.ndim != 2:
            msg = "H must be a 2D np.ndarray."
            raise ValueError(msg)

        if 0 < trial_size < 1:
            batch_size = int(trial_size * H.shape[0])
        elif trial_size >= 1 and isinstance(trial_size, int):
            batch_size = trial_size
        else:
            msg = (
                "Invalid trial_size parameter. trial_size must be either "
                "a positive integer or a float between 0 and 1."
            )
            raise ValueError(msg)

        # Throw an error if the batch size is too large or too small.
        if batch_size > H.shape[0]:
            msg = (
                "Error bagging the input data. Please ensure that the "
                "trial_size parameter is small enough for bagging."
            )
            raise ValueError(msg)

        if batch_size == 0:
            msg = (
                "Error bagging the input data. Please ensure that the "
                "trial_size parameter is large enough for bagging."
            )
            raise ValueError(msg)

        # Obtain and return subset of the data.
        all_inds = np.arange(H.shape[0])
        subset_inds = np.sort(
            np.random.choice(all_inds, size=batch_size, replace=False)
        )
        return H[subset_inds], subset_inds

    def _variable_projection(self, H, t, init_alpha, Phi, dPhi):
        """
        Variable projection routine for multivariate data.
        Attempts to fit the columns of H as linear combinations of the columns
        of Phi(alpha,t) such that H = Phi(alpha,t)B. Note that M denotes the
        number of data samples, N denotes the number of columns of Phi, IS
        denotes the number of functions to fit, and IA denotes the length
        of the alpha vector.

        :param H: (M, IS) matrix of data.
        :type H: numpy.ndarray
        :param t: (M,) vector of sample times.
        :type t: numpy.ndarray
        :param init_alpha: initial guess for alpha.
        :type init_alpha: numpy.ndarray
        :param Phi: (M, N) matrix-valued function Phi(alpha,t).
        :type Phi: function
        :param dPhi: (M, N) matrix-valued function dPhi(alpha,t,i) that
            contains the derivatives of Phi wrt the ith component of alpha.
        :type dPhi: function
        :return: Tuple of two numpy arrays and a boolean representing:
            1. (N, IS) best-fit matrix B.
            2. (N,) best-fit vector alpha.
            3. Flag indicating whether or not convergence was reached.
        :rtype: Tuple[numpy.ndarray, numpy.ndarray, bool]

        References:
        - Extensions and Uses of the Variable Projection Algorith for Solving
        Nonlinear Least Squares Problems by G. H. Golub and R. J. LeVeque ARO
        Report 79-3, Proceedings of the 1979 Army Numerical Analsysis and
        Computers Conference.
        - Variable projection for nonlinear least squares problems.
        Computational Optimization and Applications 54.3 (2013): 579-593
        by Dianne P. O'Leary and Bert W. Rust.
        """

        def compute_residual(alpha):
            """
            Helper function that, given alpha, and using H, t, Phi as they are
            passed to the _variable_projection function, computes and returns
            the matrix Phi(alpha,t), B from the expression H = Phi(alpha,t)B,
            the residual H - Phi(alpha,t)B, and 0.5*norm(residual,'fro')^2,
            which is used as an error indicator.
            """
            Phi_matrix = Phi(alpha, t)
            B = np.linalg.lstsq(Phi_matrix, H, rcond=None)[0]
            residual = H - Phi_matrix.dot(B)
            error = 0.5 * np.linalg.norm(residual) ** 2
            return B, residual, error

        # Define M, IS, and IA.
        M, IS = H.shape
        IA = len(init_alpha)

        # Unpack all variable projection parameters stored in varpro_opts.
        (
            init_lambda,
            maxlam,
            lamup,
            use_levmarq,
            maxiter,
            tol,
            eps_stall,
            use_fulljac,
            verbose,
        ) = self._varpro_opts

        # Initialize values.
        tolrank = M * np.finfo(float).eps
        _lambda = init_lambda
        alpha = self._push_eigenvalues(init_alpha)
        B, residual, error = compute_residual(alpha)
        relative_error = np.linalg.norm(residual) / np.linalg.norm(H)
        U, S, Vh = self._compute_irank_svd(Phi(alpha, t), tolrank)

        # Initialize termination flags.
        converged = False
        stalled = False

        # Initialize storage.
        all_error = np.zeros(maxiter)
        djac_matrix = np.zeros((M * IS, IA), dtype="complex")
        rjac = np.zeros((2 * IA, IA), dtype="complex")
        scales = np.zeros(IA)

        for itr in range(maxiter):
            # Build Jacobian matrix, looping over alpha indices.
            for i in range(IA):
                # Build the approximate expression for the Jacobian.
                dphi_temp = dPhi(alpha, t, i)
                ut_dphi = csr_matrix(U.conj().T @ dphi_temp)
                uut_dphi = csr_matrix(U @ ut_dphi)
                djac_a = (dphi_temp - uut_dphi) @ B
                djac_matrix[:, i] = djac_a.ravel(order="F")

                # Compute the full expression for the Jacobian.
                if use_fulljac:
                    transform = np.linalg.multi_dot([U, np.linalg.inv(S), Vh])
                    dphit_res = csr_matrix(dphi_temp.conj().T @ residual)
                    djac_b = transform @ dphit_res
                    djac_matrix[:, i] += djac_b.ravel(order="F")

                # Scale for the Levenberg algorithm.
                scales[i] = 1
                # Scale for the Levenberg-Marquardt algorithm.
                if use_levmarq:
                    scales[i] = min(np.linalg.norm(djac_matrix[:, i]), 1)
                    scales[i] = max(scales[i], 1e-6)

            # Loop to determine lambda (the step-size parameter).
            rhs_temp = np.copy(residual.ravel(order="F"))[:, None]
            q_out, djac_out, j_pvt = qr(
                djac_matrix, mode="economic", pivoting=True
            )
            ij_pvt = np.arange(IA)
            ij_pvt = ij_pvt[j_pvt]
            rjac[:IA] = np.triu(djac_out[:IA])
            rhs_top = q_out.conj().T.dot(rhs_temp)
            scales_pvt = scales[j_pvt[:IA]]
            rhs = np.concatenate(
                (rhs_top[:IA], np.zeros(IA, dtype="complex")), axis=None
            )

            def step(_lambda, rhs, scales_pvt, ij_pvt):
                """
                Helper function that, given a step size _lambda and the current
                right-hand side and pivots, computes and returns delta, the
                amount in which we update alpha, and the updated alpha vector.
                Note that this function uses rjac and alpha as they are defined
                outside of this function.
                """
                # Compute the step delta.
                rjac[IA:] = _lambda * np.diag(scales_pvt)
                delta = np.linalg.lstsq(rjac, rhs, rcond=None)[0]
                delta = delta[ij_pvt]
                # Compute the updated alpha vector.
                alpha_updated = alpha.ravel() + delta.ravel()
                alpha_updated = self._push_eigenvalues(alpha_updated)
                return delta, alpha_updated

            # Take a step using our initial step size init_lambda.
            delta_0, alpha_0 = step(_lambda, rhs, scales_pvt, ij_pvt)
            B_0, residual_0, error_0 = compute_residual(alpha_0)

            # Check actual improvement vs predicted improvement.
            actual_improvement = error - error_0
            pred_improvement = (
                0.5
                * np.linalg.multi_dot(
                    [delta_0.conj().T, djac_matrix.conj().T, rhs_temp]
                )[0].real
            )
            improvement_ratio = actual_improvement / pred_improvement

            if error_0 < error:
                # Rescale lambda based on the improvement ratio.
                _lambda *= max(1 / 3, 1 - (2 * improvement_ratio - 1) ** 3)
                alpha, B, residual, error = alpha_0, B_0, residual_0, error_0
            else:
                # Increase lambda until something works.
                for _ in range(maxlam):
                    _lambda *= lamup
                    delta_0, alpha_0 = step(_lambda, rhs, scales_pvt, ij_pvt)
                    B_0, residual_0, error_0 = compute_residual(alpha_0)

                    if error_0 < error:
                        break

                # Terminate if no appropriate step length was found...
                if error_0 >= error:
                    if verbose:
                        msg = (
                            "Failed to find appropriate step length at "
                            "iteration {}. Current error {}. "
                            "Consider increasing maxlam or lamup."
                        )
                        print(msg.format(itr + 1, relative_error))
                    return B, alpha, converged

                # ...otherwise, update and proceed.
                alpha, B, residual, error = alpha_0, B_0, residual_0, error_0

            # Record the current relative error.
            relative_error = np.linalg.norm(residual) / np.linalg.norm(H)
            all_error[itr] = relative_error

            # Print iterative progress if the verbose flag is turned on.
            if verbose:
                update_msg = "Step {} Error {} Lambda {}"
                print(update_msg.format(itr + 1, relative_error, _lambda))

            # Update termination status and terminate if converged or stalled.
            converged = relative_error < tol
            error_reduction = all_error[itr - 1] - all_error[itr]
            stalled = (itr > 0) and (
                error_reduction < eps_stall * all_error[itr - 1]
            )

            if converged:
                if verbose:
                    print("Convergence reached!")
                return B, alpha, converged

            if stalled:
                if verbose:
                    msg = (
                        "Stall detected: error reduced by less than {} "
                        "times the error at the previous step. "
                        "Iteration {}. Current error {}. Consider "
                        "increasing tol or decreasing eps_stall."
                    )
                    print(msg.format(eps_stall, itr + 1, relative_error))
                return B, alpha, converged

            U, S, Vh = self._compute_irank_svd(Phi(alpha, t), tolrank)

        # Failed to meet tolerance in maxiter steps.
        if verbose:
            msg = (
                "Failed to reach tolerance after maxiter = {} iterations. "
                "Current error {}."
            )
            print(msg.format(maxiter, relative_error))

        return B, alpha, converged

    def _single_trial_compute_operator(self, H, t, init_alpha):
        """
        Helper function that computes the standard optimized dmd operator.
        Returns the resulting DMD modes, eigenvalues, amplitudes, reduced
        system matrix, full system matrix, and whether or not convergence
        of the variable projection routine was reached.
        """
        B, alpha, converged = self._variable_projection(
            H, t, init_alpha, self._exp_function, self._exp_function_deriv
        )
        # Save the modes, eigenvalues, and amplitudes respectively.
        w = B.T
        e = alpha
        b = np.sqrt(np.sum(np.abs(w) ** 2, axis=0))

        # Normalize the modes and the amplitudes.
        inds_small = np.abs(b) < (10 * np.finfo(float).eps * np.max(b))
        b[inds_small] = 1.0
        w = w.dot(np.diag(1 / b))
        w[:, inds_small] = 0.0
        b[inds_small] = 0.0

        # Compute the projected propagator Atilde.
        if self._use_proj:
            Atilde = np.linalg.multi_dot([w, np.diag(e), np.linalg.pinv(w)])
            # Unproject the dmd modes.
            w = self._proj_basis.dot(w)
        else:
            w_proj = self._proj_basis.conj().T.dot(w)
            Atilde = np.linalg.multi_dot(
                [w_proj, np.diag(e), np.linalg.pinv(w_proj)]
            )

        # Compute the full system matrix A.
        if self._compute_A:
            A = np.linalg.multi_dot([w, np.diag(e), np.linalg.pinv(w)])
        else:
            A = None

        return w, e, b, Atilde, A, converged

    def compute_operator(self, H, t):
        """
        Compute the low-rank and the full BOP-DMD operators.

        :param H: Matrix of data to fit.
        :type H: numpy.ndarray
        :param t: Vector of sample times.
        :type t: numpy.ndarray
        :return: The BOP-DMD amplitudes.
        :rtype: numpy.ndarray
        """
        # Perform an initial optimized dmd solve using init_alpha.
        initial_optdmd_results = self._single_trial_compute_operator(
            H, t, self._init_alpha
        )
        w_0, e_0, b_0, Atilde_0, A_0, converged = initial_optdmd_results

        # Generate a warning if convergence wasn't initially reached.
        if not converged:
            msg = (
                "Initial trial of Optimized DMD failed to converge. "
                "Consider re-adjusting your variable projection parameters "
                "with the varpro_opts_dict and consider setting verbose=True."
            )
            warnings.warn(msg)

        # If num_trials isn't a positive int, perform standard optimized dmd.
        if self._num_trials <= 0 or not isinstance(self._num_trials, int):
            self._modes = w_0
            self._eigenvalues = e_0
            self._Atilde = Atilde_0
            self._A = A_0
            return b_0

        # Otherwise, perform BOP-DMD.
        verbose = self._varpro_opts[-1]
        if verbose:
            num_trial_print = 5
            msg = "\nDisplaying the results of the next {} trials...\n"
            print(msg.format(num_trial_print))

        # Initialize storage for values needed for stat computations.
        w_sum = np.zeros(w_0.shape, dtype="complex")
        e_sum = np.zeros(e_0.shape, dtype="complex")
        b_sum = np.zeros(b_0.shape, dtype="complex")
        w_sum2 = np.zeros(w_0.shape, dtype="complex")
        e_sum2 = np.zeros(e_0.shape, dtype="complex")
        b_sum2 = np.zeros(b_0.shape, dtype="complex")

        # Perform num_trials many successful trials of optimized dmd.
        num_successful_trials = 0
        num_consecutive_fails = 0
        runtime_warning_given = False

        while num_successful_trials < self._num_trials:
            H_i, subset_inds = self._bag(H, self._trial_size)
            trial_optdmd_results = self._single_trial_compute_operator(
                H_i, t[subset_inds], e_0
            )
            w_i, e_i, b_i, _, _, converged = trial_optdmd_results
            if verbose:
                print()
                num_trial_print -= 1
                verbose = num_trial_print > 0
                self._varpro_opts[-1] = verbose

            # Incorporate results into the running average
            # ONLY IF the trial successfully converged.
            if converged:
                sorted_inds = self._argsort_eigenvalues(e_i)

                # Add to iterative sums.
                w_sum += w_i[:, sorted_inds]
                e_sum += e_i[sorted_inds]
                b_sum += b_i[sorted_inds]

                # Add to iterative sums of squares.
                w_sum2 += np.abs(w_i[:, sorted_inds]) ** 2
                e_sum2 += np.abs(e_i[sorted_inds]) ** 2
                b_sum2 += np.abs(b_i[sorted_inds]) ** 2

                # Bump up the number of successful trials
                # and reset the consecutive fails counter.
                num_successful_trials += 1
                num_consecutive_fails = 0
            else:
                num_consecutive_fails += 1

            if (
                not runtime_warning_given
                and num_consecutive_fails == self._bag_warning
            ):
                msg = (
                    "{} many trials without convergence. "
                    "Consider loosening the tol requirements "
                    "of the variable projection routine."
                )
                print(msg.format(num_consecutive_fails))
                runtime_warning_given = True

            elif num_consecutive_fails == self._bag_maxfail:
                msg = (
                    "Terminating the bagging routine due to "
                    "{} many trials without convergence."
                )
                raise RuntimeError(msg.format(num_consecutive_fails))

        # Compute the BOP-DMD statistics.
        w_mu = w_sum / self._num_trials
        e_mu = e_sum / self._num_trials
        b_mu = b_sum / self._num_trials
        w_std = np.sqrt(np.abs(w_sum2 / self._num_trials - np.abs(w_mu) ** 2))
        e_std = np.sqrt(np.abs(e_sum2 / self._num_trials - np.abs(e_mu) ** 2))
        b_std = np.sqrt(np.abs(b_sum2 / self._num_trials - np.abs(b_mu) ** 2))

        # Save the BOP-DMD statistics.
        self._modes = w_mu
        self._eigenvalues = e_mu
        self._modes_std = w_std
        self._eigenvalues_std = e_std
        self._amplitudes_std = b_std

        # Compute Atilde using the average optimized dmd results.
        w_proj = self._proj_basis.conj().T.dot(self._modes)
        self._Atilde = np.linalg.multi_dot(
            [w_proj, np.diag(self._eigenvalues), np.linalg.pinv(w_proj)]
        )
        # Compute A if requested.
        if self._compute_A:
            self._A = np.linalg.multi_dot(
                [
                    self._modes,
                    np.diag(self._eigenvalues),
                    np.linalg.pinv(self._modes),
                ]
            )

        return b_mu


class BOPDMD(DMDBase):
    """
    Bagging, Optimized Dynamic Mode Decomposition.

    :param svd_rank: The rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive integer, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation.
    :type svd_rank: int or float
    :param compute_A: Flag that determines whether or not to compute the full
        Koopman operator A. Default is False, do not compute the full operator.
        Note that the full operator is potentially prohibitively expensive to
        compute.
    :type compute_A: bool
    :param use_proj: Flag that determines the type of computation to perform.
        If True, fit input data projected onto the first svd_rank POD modes or
        columns of proj_basis if provided. If False, fit the full input data.
        Default is True, fit projected data.
    :type use_proj: bool
    :param init_alpha: Initial guess for the continuous-time DMD eigenvalues.
        If not provided, one is computed via a trapezoidal rule approximation.
        Default is None (alpha not provided).
    :type init_alpha: numpy.ndarray
    :param proj_basis: Orthogonal basis for projection, where each column of
        proj_basis contains a basis mode. If not provided, POD modes are used.
        Default is None (basis not provided).
    :type proj_basis: numpy.ndarray
    :param num_trials: Number of BOP-DMD trials to perform. If num_trials is a
        positive integer, num_trials BOP-DMD trials are performed. Otherwise,
        standard optimized dmd is performed. Default is 0.
    :type num_trials: int
    :param trial_size: Size of the randomly selected subset of observations to
        use for each trial of bagged optimized dmd (BOP-DMD). If trial_size is
        a positive integer, trial_size many observations will be used per
        trial. If trial_size is a float between 0 and 1, int(trial_size * m)
        many observations will be used per trial, where m denotes the total
        number of data points observed. Note that any other type of input for
        trial_size will yield an error. Default is 0.2.
    :type trial_size: int or float
    :param eig_sort: Method used to sort eigenvalues (and modes accordingly)
        when performing BOP-DMD. Eigenvalues will be sorted by real part and
        then by imaginary part to break ties if `eig_sort="real"`, by imaginary
        part and then by real part to break ties if `eig_sort="imag"`, or by
        magnitude if `eig_sort="abs"`. If `eig_sort="auto"`, one of the
        previously-mentioned sorting methods is chosen depending on eigenvalue
        variance. Default is "auto".
    :type eig_sort: {"real", "imag", "abs", "auto"}
    :param eig_constraints: Set containing desired DMD operator eigenvalue
        constraints. Currently available constraints are `"stable"`, which
        constrains eigenvalues to the left half of the complex plane,
        `"imag"`, which constrains eigenvalues to the imaginary axis, and
        `"conjugate_pairs"`, which enforces that eigenvalues are always
        present with their complex conjugate. Note that constraints may be
        combined if valid. May alternatively be a custom eigenvalue constraint
        function that will be applied to the computed eigenvalues at each step
        of the variable projection routine.
    :type eig_constraints: set(str) or function
    :param bag_warning: Number of consecutive non-converged trials of BOP-DMD
        at which to produce a warning message for the user. Default is 100.
        Use arguments less than or equal to zero for no warning condition.
    :type bag_warning: int
    :param bag_maxfail: Number of consecutive non-converged trials of BOP-DMD
        at which to terminate the fit. Default is -1, no stopping condition.
    :type bag_maxfail: int
    :param varpro_opts_dict: Dictionary containing the desired parameter values
        for variable projection. The following parameters may be specified:
        `init_lambda`, `maxlam`, `lamup`, `use_levmarq`, `maxiter`, `tol`,
        `eps_stall`, `use_fulljac`, `verbose`. Default values will be used for
        any parameters not specified in `varpro_opts_dict`.
        See `BOPDMDOperator` documentation for default values and descriptions
        for each parameter.
    :type varpro_opts_dict: dict
    """

    def __init__(
        self,
        svd_rank=0,
        compute_A=False,
        use_proj=True,
        init_alpha=None,
        proj_basis=None,
        num_trials=0,
        trial_size=0.6,
        eig_sort="auto",
        eig_constraints=None,
        bag_warning=100,
        bag_maxfail=-1,
        varpro_opts_dict=None,
    ):
        self._svd_rank = svd_rank
        self._compute_A = compute_A
        self._use_proj = use_proj
        self._init_alpha = init_alpha
        self._proj_basis = proj_basis
        self._num_trials = num_trials
        self._trial_size = trial_size
        self._eig_sort = eig_sort

        if not isinstance(bag_warning, int) or not isinstance(bag_maxfail, int):
            msg = (
                "bag_warning and bag_maxfail must be integers. "
                "Please use a non-positive integer if no warning "
                "or stopping condition is desired."
            )
            raise TypeError(msg)
        self._bag_warning = bag_warning
        self._bag_maxfail = bag_maxfail

        if varpro_opts_dict is None:
            self._varpro_opts_dict = {}
        elif not isinstance(varpro_opts_dict, dict):
            raise TypeError("varpro_opts_dict must be a dict.")
        else:
            self._varpro_opts_dict = varpro_opts_dict

        if eig_constraints is None:
            eig_constraints = set()
        elif not isinstance(eig_constraints, set) and not isfunction(
            eig_constraints
        ):
            raise TypeError("eig_constraints must be a set or a function.")
        self._check_eig_constraints(eig_constraints)
        self._eig_constraints = eig_constraints

        self._snapshots_holder = None
        self._time = None
        self._Atilde = None
        self._modes_activation_bitmask_proxy = None

    @property
    def svd_rank(self):
        """
        :return: the rank used for the svd truncation.
        :rtype: int or float
        """
        return self._svd_rank

    @property
    def compute_A(self):
        """
        :return: flag that determines whether to compute the full operator A.
        :rtype: bool
        """
        return self._compute_A

    @property
    def use_proj(self):
        """
        :return: flag that determines whether to fit projected or full data.
        :rtype: bool
        """
        return self._use_proj

    @property
    def init_alpha(self):
        """
        :return: initial guess used for the continuous-time DMD eigenvalues.
        :rtype: numpy.ndarray
        """
        if self._init_alpha is None:
            msg = (
                "fit() hasn't been called "
                "and no initial value for alpha has been given."
            )
            raise RuntimeError(msg)
        return self._init_alpha

    @property
    def proj_basis(self):
        """
        :return: the projection basis used, with modes stored by column.
        :rtype: numpy.ndarray
        """
        if self._proj_basis is None:
            msg = (
                "fit() hasn't been called "
                "and no projection basis has been given."
            )
            raise RuntimeError(msg)
        return self._proj_basis

    @property
    def num_trials(self):
        """
        :return: the number of BOP-DMD trials to perform.
        :rtype: int
        """
        return self._num_trials

    @property
    def trial_size(self):
        """
        :return: size of the data subsets used during each BOP-DMD trial.
        :rtype: int or float
        """
        return self._trial_size

    @property
    def time(self):
        """
        Get the vector that contains the time points of the fitted snapshots.

        :return: the vector that contains the original time points.
        :rtype: numpy.ndarray
        """
        if self._time is None:
            raise RuntimeError("fit() hasn't been called.")
        return self._time

    @property
    def atilde(self):
        """
        Get the reduced Koopman operator A, called Atilde.

        :return: the reduced Koopman operator A.
        :rtype: numpy.ndarray
        """
        return self.operator.as_numpy_array

    @property
    def A(self):
        """
        Get the full Koopman operator A.

        :return: the full Koopman operator A.
        :rtype: numpy.ndarray
        """
        return self.operator.A

    @property
    def dynamics(self):
        """
        Get the time evolution of each mode.

        :return: matrix that contains all the time evolution, stored by row.
        :rtype: numpy.ndarray
        """
        t_omega = np.exp(np.outer(self.eigs, self._time))
        return np.diag(self.amplitudes).dot(t_omega)

    @property
    def amplitudes_std(self):
        """
        Get the amplitudes standard deviation.

        :return: amplitudes standard deviation.
        :rtype: numpy.ndarray
        """
        return self.operator.amplitudes_std

    @property
    def eigenvalues_std(self):
        """
        Get the eigenvalues standard deviation.

        :return: eigenvalues standard deviation.
        :rtype: numpy.ndarray
        """
        return self.operator.eigenvalues_std

    @property
    def modes_std(self):
        """
        Get the modes standard deviation.

        :return: modes standard deviation.
        :rtype: numpy.ndarray
        """
        return self.operator.modes_std

    @property
    def eig_constraints(self):
        """
        Get the eigenvalue constraints.

        :return: eigenvalue constraints.
        :rtype: set(str)
        """
        return self._eig_constraints

    def print_varpro_opts(self):
        """
        Prints a formatted information string that displays all chosen
        variable projection parameter values.
        """
        if self._Atilde is None:
            raise ValueError("You need to call fit before")

        opt_names = [
            "init_lambda",
            "maxlam",
            "lamup",
            "use_levmarq",
            "maxiter",
            "tol",
            "eps_stall",
            "use_fulljac",
            "verbose",
        ]
        print("VARIABLE PROJECTION OPTIONS:")
        print("============================")
        for name, value in zip(opt_names, self.operator.varpro_opts):
            if len(name) < 7:
                print(name + ":\t\t" + str(value))
            else:
                print(name + ":\t" + str(value))

    def _check_eig_constraints(self, eig_constraints):
        """
        Function that verifies that...
         - if the given eig_constraints is a function, the function is able to
         take a (n,) numpy.ndarray and return a (n,) numpy.ndarray
         - if the given eig_constraints is a set, it does not contain an
         unsupported constraint class, and does not contain an invalid
         combination of eigenvalue constraints
        """
        if isfunction(eig_constraints):
            if self._svd_rank >= 1 and isinstance(self._svd_rank, int):
                r = self._svd_rank
            else:  # use a dummy rank of 10
                r = 10

            try:
                # Test the eig_constraints function on a (r,) np.ndarray.
                x_dummy = np.arange(r)
                y_dummy = eig_constraints(x_dummy)
            except Exception as e:
                msg = (
                    "eigenvalue constraint function must be able "
                    "to take a single (n,) numpy.ndarray as input."
                )
                raise ValueError(msg) from e

            if not isinstance(y_dummy, np.ndarray):
                msg = (
                    "eigenvalue constraint function must "
                    "output a single (n,) numpy.ndarray."
                )
                raise ValueError(msg)

            if x_dummy.shape != y_dummy.shape:
                msg = (
                    "eigenvalue constraint function must accept a (n,) "
                    "numpy.ndarray as input and output a (n,) numpy.ndarray."
                )
                raise ValueError(msg)

        else:
            valid_constraints = {"stable", "imag", "conjugate_pairs"}
            invalid_combos = [{"stable", "imag"}]

            if len(eig_constraints.difference(valid_constraints)) != 0:
                raise ValueError("Invalid eigenvalue constraint provided.")

            for invalid_combo_set in invalid_combos:
                if invalid_combo_set.issubset(eig_constraints):
                    msg = "Invalid eigenvalue constraint combination provided."
                    raise ValueError(msg)

    def _initialize_alpha(self):
        """
        Uses projected trapezoidal rule to approximate the eigenvalues of A in
            z' = Az.
        The computed eigenvalues will serve as our initial guess for alpha.

        :return: Approximated eigenvalues of the matrix A.
        :rtype: numpy.ndarray
        """
        # Project the snapshot data onto the projection basis.
        ux = self._proj_basis.conj().T.dot(self.snapshots)
        ux1 = ux[:, :-1]
        ux2 = ux[:, 1:]

        # Define the diagonal matrix T as the following.
        t1 = self._time[:-1]
        t2 = self._time[1:]
        T = np.diag(t2 - t1)

        # Define the matrices Y and Z as the following and compute the
        # rank-truncated SVD of Y.
        Y = (ux1 + ux2) / 2
        Z = (ux2 - ux1).dot(np.linalg.inv(T))
        U, s, V = compute_svd(Y, self._svd_rank)
        S = np.diag(s)

        # Compute the matrix Atilde and return its eigenvalues.
        Atilde = np.linalg.multi_dot([U.conj().T, Z, V, np.linalg.inv(S)])

        return np.linalg.eig(Atilde)[0]

    def fit(self, X, t):
        """
        Compute the Optimized Dynamic Mode Decomposition.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        :param t: the input time vector.
        :type t: numpy.ndarray or iterable
        """
        # Process the input data and convert to numpy.ndarrays.
        self._reset()
        self._snapshots_holder = Snapshots(X)
        self._time = np.array(t).squeeze()

        # Check that input time vector is one-dimensional.
        if self._time.ndim > 1:
            raise ValueError("Input time vector t must be one-dimensional.")

        # Check that the number of snapshots in the data matrix X matches the
        # number of time points in the time vector t.
        if self.snapshots.shape[1] != len(self._time):
            msg = (
                "The number of columns in the data matrix X must match "
                "the number of time points in the time vector t."
            )
            raise ValueError(msg)

        # Compute the rank of the fit.
        self._svd_rank = int(compute_rank(self.snapshots, self._svd_rank))

        # Set/check the projection basis.
        if self._proj_basis is None and self._use_proj:
            self._proj_basis = compute_svd(self.snapshots, self._svd_rank)[0]
        elif self._proj_basis is None and not self._use_proj:
            self._proj_basis = compute_svd(self.snapshots, -1)[0]
        elif (
            not isinstance(self._proj_basis, np.ndarray)
            or self._proj_basis.ndim != 2
            or self._proj_basis.shape[1] != self._svd_rank
        ):
            msg = "proj_basis must be a 2D np.ndarray with {} columns."
            raise ValueError(msg.format(self._svd_rank))

        # Set/check the initial guess for the continuous-time DMD eigenvalues.
        if self._init_alpha is None:
            self._init_alpha = self._initialize_alpha()
        elif (
            not isinstance(self._init_alpha, np.ndarray)
            or self._init_alpha.ndim > 1
            or len(self._init_alpha) != self._svd_rank
        ):
            msg = "init_alpha must be a 1D np.ndarray with {} entries."
            raise ValueError(msg.format(self._svd_rank))

        # Build the BOP-DMD operator now that the initial alpha and
        # the projection basis have been defined.
        self._Atilde = BOPDMDOperator(
            self._compute_A,
            self._use_proj,
            self._init_alpha,
            self._proj_basis,
            self._num_trials,
            self._trial_size,
            self._eig_sort,
            self._eig_constraints,
            self._bag_warning,
            self._bag_maxfail,
            **self._varpro_opts_dict
        )

        # Define the snapshots that will be used for fitting.
        if self._use_proj:
            snp = self._proj_basis.conj().T.dot(self.snapshots)
        else:
            snp = self.snapshots

        # Fit the data.
        self._b = self.operator.compute_operator(snp.T, self._time)

        return self

    def forecast(self, t):
        """
        Predict the output X given the input time t using the fitted DMD model.
        If model has been fitted using multiple ensembles, an average
        prediction and its variance is returned.

        :param t: the input time vector.
        :type t: numpy.ndarray or iterable
        :return: system prediction at times given by vector t.
        :rtype: numpy.ndarray or numpy.ndarray, numpy.ndarray
        """
        # Process the input data and convert to numpy.ndarray.
        t = np.array(t).squeeze()

        # Reject the input time vector if it isn't one-dimensional.
        if t.ndim > 1:
            raise ValueError("Input time vector t must be one-dimensional.")

        # If variance information has been recorded, use it.
        if self.operator.eigenvalues_std is not None:
            # Compute num_trials many forecasts.
            all_x = np.empty(
                (self._num_trials, self.snapshots.shape[0], len(t)),
                dtype="complex",
            )

            for k in range(self._num_trials):
                # Draw eigenvalues and amplitudes from random distribution.
                eigs_k = self.eigs + np.multiply(
                    np.random.randn(*self.eigs.shape),
                    self.operator.eigenvalues_std,
                )
                b_k = self.amplitudes + np.multiply(
                    np.random.randn(*self.amplitudes.shape),
                    self.operator.amplitudes_std,
                )
                # Compute forecast using average modes and eigs_k, b_k.
                all_x[k] = np.linalg.multi_dot(
                    [self.modes, np.diag(b_k), np.exp(np.outer(eigs_k, t))]
                )

            # Return the average forecast and the variance.
            return np.mean(all_x, axis=0), np.var(all_x, axis=0)

        # If no variance information, simply compute a standard forecast.
        x = np.linalg.multi_dot(
            [
                self.modes,
                np.diag(self.amplitudes),
                np.exp(np.outer(self.eigs, t)),
            ]
        )
        return x
