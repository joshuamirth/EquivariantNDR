import autograd.numpy as np
import autograd.numpy.linalg as LA
import pymanopt
from pymanopt.manifolds import Oblique  # Change this one.
from pymanopt.solvers import *

#def gmds():

def setup_grassmannian_cost(D,return_derivatives=False):
    def F(Q):
        return 0.5*LA.norm(grassmann_distance_matrix_basis(Q) - D)**2
    def dF(Q):
        return ???
    if return_derivatives:
        return F, dF
    else:
        return F

def grassmann_distance_matrix_basis(Q, metric='geodesic'):
    """Matrix of principal angles between subspaces
    
    Parameters
    ----------
    Q : ndarray (N*n*k)
        Collection of `N` `k`-dimensional subspaces of
        :math:`\mathbb{R}^n` represented by `(n*k)`-matrices giving
        orthonormal bases for each subspace "stacked" into a 3-D array.
    metric : string, ('geodesic', 'chordal', 'min, 'gap')
        Choice of (pseudo-) metric on the Grassmannian. The geodesic
        distance is given by the 2-norm of the principal angle vector,
        the chordal distance is the 2-norm of the sine of the principal
        angle vector, `'min'` gives the first (smallest) principal
        angle, and `'gap'` gives the sine of the last (largest)
        principal angle.

    Returns
    -------
    Theta : ndarray (N*N)
        Matrix of distances between subspaces.

    Examples
    --------
    >>> Q = np.stack((Q1,Q2,Q3),axis=0)


    """

    M = np.tensordot(Q,Q,axes=(1,1))
    M = np.swapaxes(M,1,2)  # Want first two axes to index inner product.
    _,S,_ = LA.svd(M)
    if metric == 'geodesic':
        Theta = LA.norm(np.arccos(acos_validate(S)),axis=2)
    elif metric == 'min':
        Theta = np.min(np.arccos(acos_validate(S)),axis=2)
    else:
        raise NotImplementedError('Metric %s not yet implemented. Only '\
            'geodesic and min currently available.' %metric)
    return Theta
    
#def grassmann_distance_matrix_projection()
# Represent points on the Grassmannian as either 
#    """Matrix of principal angles between subspaces
#    
#    Parameters
##    ----------
#    P : ndarray (N*n*n)
#        Collection of `N` `k`-dimensional subspaces of
#        :math:`\mathbb{R}^n` represented by `(n*n)` projection matrices
#        for each subspace "stacked" into a 3-D array.
#
#    """

def acos_validate(M):
    """Replace values outside the domain of arccosine with +/- 1.

    Parameters
    ----------
    M : ndarray
        Array of values approximately between -1 and 1.

    Returns
    -------
    M : ndarray
        Original array with values greater than 1 replaced by 1 and
        values less than -1 replaced by -1.

    Notes
    -----
    The input array is modified in place.

    """
    big_vals = M >= 1.0
    M[big_vals] = 1.0
    small_vals = M <= -1.0
    M[small_vals] = -1.0
    return M

