"""
Derived module from dmd.py for Physics-informed DMD.

References:
- Peter J. Baddoo, Benjamin Herrmann, Beverley J. McKeon, J. Nathan Kutz, and
Steven L. Brunton. Physics-informed dynamic mode decomposition (pidmd). 2021.
arXiv:2112.04307.
"""
import numpy as np

from .dmd import DMD
from .dmdoperator import DMDOperator
from .utils import compute_svd
from .pidmd_utils import (
    compute_unitary,
    compute_uppertriangular,
    compute_diagonal,
    compute_symmetric,
    compute_toeplitz,
    compute_circulant,
    compute_symtridiagonal,
    compute_BCCB,
    compute_BC,
)


class PiDMDOperator(DMDOperator):
    """
    DMD operator for Physics-informed DMD.

    :param manifold: the matrix manifold for the full DMD operator A.
    :type manifold: str
    :param manifold_opt: option used to specify certain manifolds. If manifold
        is "diagonal", manifold_opt may be used to specify the width of the
        diagonal of A. If manifold starts with "BC", manifold_opt must be a
        2D tuple that specifies the desired dimensions of the blocks of A.
    :type manifold_opt: int, tuple(int,int), or numpy.ndarray
    :param compute_A: Flag that determines whether or not to compute the full
        Koopman operator A.
    :type compute_A: bool
    :param svd_rank: the rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive integer, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation.
    :type svd_rank: int or float
    """

    def __init__(
        self,
        manifold,
        manifold_opt,
        compute_A,
        svd_rank,
    ):
        self._manifold = manifold
        self._manifold_opt = manifold_opt
        self._compute_A = compute_A
        self._svd_rank = svd_rank

        self._A = None
        self._Atilde = None
        self._eigenvalues = None
        self._eigenvectors = None
        self._modes = None

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

    def _check_compute_A(self):
        """
        Helper method that checks that compute_A is True.
        Throws an error if compute_A is False.
        """
        if not self._compute_A:
            raise ValueError(
                "A must be computed for the chosen manifold."
                "Set compute_A = True to compute A."
            )

    def _compute_procrustes(self, X, Y):
        """
        Private method that computes the best-fit linear operator A in the
        relationship Y = AX such that A is restricted to the family of matrices
        defined by the given manifold (and manifold option if applicable).
        Computes and returns a dictionary that contains either...
        (1) the reduced operator "atilde",
        (2) the full operator "A", or
        (3) the "eigenvalues" and eigenvectors of A, referred to as "modes",
        depending on the chosen manifold and the compute_A parameter.
        """
        if self._manifold == "unitary":
            result_dict = compute_unitary(X, Y, self._svd_rank)
        elif self._manifold == "uppertriangular":
            self._check_compute_A()
            result_dict = compute_uppertriangular(X, Y)
        elif self._manifold == "lowertriangular":
            self._check_compute_A()
            A_rot = compute_uppertriangular(np.flipud(X), np.flipud(Y))["A"]
            result_dict = {"A": np.rot90(A_rot, 2)}
        elif self._manifold == "diagonal":
            result_dict = compute_diagonal(
                X, Y, self._svd_rank, self._manifold_opt, self._compute_A
            )
        elif self._manifold == "symmetric":
            result_dict = compute_symmetric(X, Y, self._svd_rank)
        elif self._manifold == "skewsymmetric":
            result_dict = compute_symmetric(
                X, Y, self._svd_rank, skew_symmetric=True
            )
        elif self._manifold == "toeplitz":
            self._check_compute_A()
            result_dict = compute_toeplitz(X, Y)
        elif self._manifold == "hankel":
            self._check_compute_A()
            result_dict = compute_toeplitz(X, Y, flipped=True)
        elif self._manifold in [
            "circulant",
            "circulant_unitary",
            "circulant_symmetric",
            "circulant_skewsymmetric",
        ]:
            circulant_opt = self._manifold.replace("circulant_", "")
            result_dict = compute_circulant(
                X, Y, circulant_opt, self._svd_rank, self._compute_A
            )
        elif self._manifold == "symmetric_tridiagonal":
            result_dict = compute_symtridiagonal(
                X, Y, self._svd_rank, self._compute_A
            )
        elif self._manifold in [
            "BC",
            "BCTB",
            "BCCB",
            "BCCBunitary",
            "BCCBsymmetric",
            "BCCBskewsymmetric",
        ]:
            # Specify the shape of the blocks in the output matrix A.
            self._check_compute_A()
            if self._manifold_opt is None:
                raise ValueError("manifold_opt must be specified.")
            if (
                not isinstance(self._manifold_opt, tuple)
                or len(self._manifold_opt) != 2
            ):
                raise ValueError("manifold_opt is not in an allowable format.")
            block_shape = np.array(self._manifold_opt)

            if self._manifold.startswith("BCCB"):
                bccb_opt = self._manifold.replace("BCCB", "")
                result_dict = compute_BCCB(
                    X, Y, block_shape, bccb_opt, self._svd_rank
                )
            elif self._manifold == "BC":
                result_dict = compute_BC(X, Y, block_shape)
            else:
                result_dict = compute_BC(
                    X, Y, block_shape, tridiagonal_blocks=True
                )
        else:
            raise ValueError("The selected manifold is not implemented.")

        return result_dict

    def compute_operator(self, X, Y):
        """
        Compute and store the low-rank operator and the full operator A.

        :param X: matrix containing the snapshots x0,..x{n-1} by column.
        :type X: numpy.ndarray
        :param Y: matrix containing the snapshots x1,..x{n} by column.
        :type Y: numpy.ndarray
        :return: the (truncated) left-singular vectors matrix, the (truncated)
            singular values array, and the (truncated) right-singular vectors
            matrix of X.
        :rtype: numpy.ndarray, numpy.ndarray, numpy.ndarray
        """
        U, s, V = compute_svd(X, self._svd_rank)

        # Compute the corresponding Procrustes problem.
        result_dict = self._compute_procrustes(X, Y)

        # Case 1: atilde was computed.
        if "atilde" in result_dict.keys():
            self._Atilde = result_dict["atilde"]
            self._eigenvalues, self._eigenvectors = np.linalg.eig(self._Atilde)
            self._modes = U.dot(self._eigenvectors)
            if self._compute_A:
                self._A = np.linalg.multi_dot(
                    [
                        self._modes,
                        np.diag(self._eigenvalues),
                        np.linalg.pinv(self._modes),
                    ]
                )
        else:  # Cases 2 and 3.
            # Case 2: A was computed.
            if "A" in result_dict.keys():
                self._A = result_dict["A"]
                self._eigenvalues, self._modes = np.linalg.eig(self._A)
            else:
                # Case 3: eigenvalues and modes were computed.
                self._eigenvalues = result_dict["eigenvalues"]
                self._modes = result_dict["modes"]
            self._eigenvectors = U.conj().T.dot(self._modes)
            self._Atilde = np.linalg.multi_dot(
                [
                    self._eigenvectors,
                    np.diag(self._eigenvalues),
                    np.linalg.pinv(self._eigenvectors),
                ]
            )

        return U, s, V


class PiDMD(DMD):
    """
    Physics-informed Dynamic Mode Decomposition.

    :param manifold: the matrix manifold to restrict the full operator A to.
        The following matrix manifolds are permissible:
        - "unitary"
        - "uppertriangular"
        - "lowertriangular"
        - "diagonal"
        - "symmetric",
        - "skewsymmetric"
        - "toeplitz"
        - "hankel"
        - "circulant"
        - "circulant_unitary"
        - "circulant_symmetric"
        - "circulant_skewsymmetric"
        - "symmetric_tridiagonal"
        - "BC" (block circulant)
        - "BCTB" (BC with tridiagonal blocks)
        - "BCCB" (BC with circulant blocks)
        - "BCCBunitary" (BCCB and unitary)
        - "BCCBsymmetric" (BCCB and symmetric)
        - "BCCBskewsymmetric" (BCCB and skewsymmetric)
    :type manifold: str
    :param manifold_opt: option used to specify certain manifolds.
        If manifold is "diagonal", manifold_opt may be used to specify the
        width of the diagonal of A. If manifold_opt is an integer k, A is
        banded, with a lower and upper bandwidth of k-1. If manifold_opt is
        a tuple containing two integers k1 and k2, A is banded with a lower
        bandwidth of k1-1 and an upper bandwidth of k2-1. Finally, if
        manifold_opt is a numpy array of size (len(X), 2), the entries of
        manifold_opt are used to explicitly define the the upper and lower
        bounds of the indices of the non-zero elements of A.
        If manifold starts with "BC", manifold_opt must be a 2D tuple that
        specifies the desired dimensions of the blocks of A.
        Note that all other manifolds do not use manifold_opt.
    :type manifold_opt: int, tuple(int,int), or numpy.ndarray
    :param compute_A: Flag that determines whether or not to compute the full
        Koopman operator A.
    :type compute_A: bool
    :param svd_rank: the rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive integer, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation. Default is -1, meaning no truncation.
    :type svd_rank: int or float
    :param tlsq_rank: rank truncation computing Total Least Square. Default is
        0, meaning no truncation.
    :type tlsq_rank: int
    :param opt: If True, amplitudes are computed like in optimized DMD  (see
        :func:`~dmdbase.DMDBase._compute_amplitudes` for reference). If
        False, amplitudes are computed following the standard algorithm. If
        `opt` is an integer, it is used as the (temporal) index of the snapshot
        used to compute DMD modes amplitudes (following the standard
        algorithm).
        The reconstruction will generally be better in time instants near the
        chosen snapshot; however increasing `opt` may lead to wrong results
        when the system presents small eigenvalues. For this reason a manual
        selection of the number of eigenvalues considered for the analyisis may
        be needed (check `svd_rank`). Also setting `svd_rank` to a value
        between 0 and 1 may give better results. Default is False.
    :type opt: bool or int
    """

    def __init__(
        self,
        manifold,
        manifold_opt=None,
        compute_A=False,
        svd_rank=-1,
        tlsq_rank=0,
        opt=False,
    ):
        super().__init__(
            svd_rank=svd_rank,
            tlsq_rank=tlsq_rank,
            opt=opt,
        )
        self._Atilde = PiDMDOperator(
            manifold=manifold,
            manifold_opt=manifold_opt,
            compute_A=compute_A,
            svd_rank=svd_rank,
        )

    @property
    def A(self):
        """
        Get the full Koopman operator A.

        :return: the full Koopman operator A.
        :rtype: numpy.ndarray
        """
        return self.operator.A
