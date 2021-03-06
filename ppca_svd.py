"""Projective PCA"""

import numpy as np 
import numpy.linalg as linalg
import time

def batch_ppca(X, dim, verbose=False):
    Xret = ppca(X, dim, batch=True, verbose=verbose)
    return Xret

def iter_ppca(X, dim, verbose=False):
    Xret = ppca(X, dim, batch=False, verbose=verbose)
    return Xret

def ppca(X, proj_dim, batch=False, verbose=False, return_basis=False):
    """
    Principal Projective Component Analysis (Jose Perea 2017)
    Parameters
    ----------
    X : ndarray (N, d)
        For all N points of the dataset, membership weights to
        d different classes are the coordinates
    proj_dim : integer
        The dimension of the ambient space onto which to project. (Dimension of projective space is proj_dim-1)
    verbose : boolean
        Whether to print information during iterations
    Returns
    -------
     'X': ndarray(N, proj_dim+1)
        The projective coordinates
     }
    """
    if verbose:
        print("Doing ppca on %i points in %i dimensions down to %i dimensions"%\
                (X.shape[0], X.shape[1], proj_dim))
    n_dim = X.shape[1]
    if proj_dim >= n_dim:
        raise ValueError('Goal dimension must be lower than original dimension.')
    tic = time.time()
    # Projective dimensionality reduction : Main Loop
    U, S, Vt = linalg.svd(X, full_matrices=False)
    if batch:
        Xret = U[:, 0:proj_dim] * S[0:proj_dim]
        Xret = (Xret.T / np.linalg.norm(Xret, axis=1)).T
        Umat = U[:, proj_dim:n_dim+1]
    else:
        Umat = np.zeros(X.shape[0])
        for i in range(n_dim-proj_dim):
            Xret = U[:, 0:-1] * S[0:-1]
            Xret = (Xret.T / np.linalg.norm(Xret, axis=1)).T
            Umat = np.column_stack((U[:,-1], Umat))
            U, S, Vt = linalg.svd(Xret, full_matrices=False)
    if verbose:
        print("Elapsed time ppca: %.3g"%(time.time()-tic))
    if return_basis:
        return Xret, Umat
    else:
        return Xret
