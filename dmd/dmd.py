import numpy as np


def dmd(X, k=None):
	"""
	Dynamic Mode Decomposition

	This method decomposes

	:param numpy.ndarray X: the input matrix with dimension `m`x`n`
	:param int k: rank truncation in SVD
	"""

	# Function aliases
	svd = lambda X: np.linalg.svd(X, full_matrices=False)
	eig = np.linalg.eig

	X1 = X[:, :-1]
	X2 = X[:, 1:]

	#---------------------------------------------------------------------------
	# Singular Value Decomposition
	#---------------------------------------------------------------------------
	U, s, V = svd(X1)
	V = np.conjugate(V.T)

	if k is not None:
		U = U[:, 0:k]
		V = V[:, 0:k]
		s = s[0:k]

	Sinverse = np.diag(1. / s)

	#---------------------------------------------------------------------------
	# DMD Modes
	#---------------------------------------------------------------------------
	X2VSinverse = X2.dot(V).dot(Sinverse)

	Atilde = np.transpose(U).dot(X2VSinverse)
	eigs, W = eig(Atilde)

	Phi = X2VSinverse.dot(W)

	#---------------------------------------------------------------------------
	# DMD Amplitudes and Dynamics
	#---------------------------------------------------------------------------
	b = np.linalg.lstsq(Phi, X1[:, 0])[0]
	B = np.diag(b)

	V = np.fliplr(np.vander(eigs, N=X.shape[1]))

	return Phi, B, V
