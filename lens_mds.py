""" Dimensionality reduction on Lens spaces using an MDS-type method. """
import autograd.numpy as np
import autograd.numpy.linalg as LA
from autograd.numpy.linalg import matrix_power as mp
import pymanopt
from pymanopt.solvers import *
from pymanopt.manifolds import Oblique

from ripser import ripser   # For classifying maps.

# dreimac does not install properly on my system.
try:
    from dreimac.projectivecoords import ppca
except:
    from ppca import ppca
    print('Loading personal version of PPCA. This may not be consistent with '\
        'the published version.')

###############################################################################
# Lens MDS Algorithm
###############################################################################

def lmds(
    Y,
    D,
    p,
    max_iter = 20,
    verbosity = 1,
    pmo_solve = 'cg',
    autograd = False,
    appx = False,
    minstepsize=1e-10,
    mingradnorm=1e-6,
    singlestep = False
):
    """Lens space multi-dimensional scaling algorithm.

    Attempts to align a collection of data points in the lens space
    :math:`L_p^n` so that the collection of distances between each pair
    of points matches a given input distance matrix as closely as
    possible.

    Parameters
    ----------
    Y : ndarray (m*d)
        Initial guess of data points. Each row corresponds to a point on
        the (2n-1)-sphere, so must have norm one and d must be even.
    D : ndarray (square)
        Distance matrix to optimize toward.
    p : int
        Cyclic group with which to act.
    max_iter: int, optional
        Maximum number of times to iterate the loop.
    verbosity: int, optional
        Amount of output to display at each iteration.
    pmo_solve: string, {'cg','sd','tr','nm','ps'}
        Solver to use with pymanopt. Default is conjugate gradient.

    Returns
    -------
    X : ndarray
        Optimal configuration of points on lens space.
    C : list (float)
        Computed cost at each loop of the iteration.
    T : list (float)
        Actual cost at each loop of the iteration.

    Notes
    -----
    The optimization can only be carried out w/r/t/ an approximation of
    the true cost function (which is not differentiable). The computed
    cost C should not match T, but should decrease when T does.

    """

    m = Y.shape[0]
    d = Y.shape[1] - 1
    W = distance_to_weights(D)
    omega = g_action_matrix(p,d)
    if appx:
        M = get_blurred_masks(Y.T,omega,p,D)
    else:
        S = optimal_rotation(Y.T,omega,p)
        M = get_masks(S,p)
    C = np.cos(D)
#   # TODO: verify that input is valid.
    cost = setup_sum_cost(omega,M,D,W,p)
    cost_list = [cost(Y.T)]
    manifold = Oblique(d+1,m) # Short, wide matrices.
    if pmo_solve == 'cg':
        if singlestep:
            solver = ConjugateGradient(
                    minstepsize=minstepsize,
                    mingradnorm=mingradnorm,
                    maxiter=2)
        else:
            solver = ConjugateGradient(
                    minstepsize=minstepsize,
                    mingradnorm=mingradnorm)
    elif pmo_solve == 'sd':
        if singlestep:
            solver = SteepestDescent(
                    minstepsize=minstepsize,
                    mingradnorm=mingradnorm,
                    maxiter=2)
        else:
            solver = SteepestDescent(
                    minstepsize=minstepsize,
                    mingradnorm=mingradnorm)
    elif pmo_solve == 'tr':
        if singlestep:
            solver = TrustRegions(
                    minstepsize=minstepsize,
                    mingradnorm=mingradnorm,
                    maxiter=2)
        else:
            solver = TrustRegions(
                    minstepsize=minstepsize,
                    mingradnorm=mingradnorm)

    elif pmo_solve == 'nm':
        solver = NelderMead()
    for i in range(0,max_iter):
        if autograd:
            cost = setup_sum_cost(omega,M,D,W,p)
            problem = pymanopt.Problem(manifold, cost, verbosity=verbosity)        
        else:
            cost, egrad = setup_sum_cost(omega,M,D,W,p,return_derivatives=True)
            problem = pymanopt.Problem(
                manifold,
                cost,
                egrad = egrad,
                verbosity = verbosity
            )
        if pmo_solve == 'cg' or pmo_solve == 'sd' or pmo_solve == 'tr':
            Y_new = solver.solve(problem, x=Y.T)
        else:
            Y_new =  solver.solve(problem)
        Y_new = Y_new.T     # Y should be tall-skinny
        cost_oldM = cost(Y_new.T)
        # cost_list.append(cost_oldM)
        if appx:
            M_new = get_blurred_masks(Y_new.T,omega,p,D)
        else:
            S_new = optimal_rotation(Y_new.T,omega,p)
            M_new = get_masks(S_new,p)
        cost_new = setup_sum_cost(omega,M_new,D,W,p)
        cost_newM = cost_new(Y_new.T)
        cost_list.append(cost_newM)
#       S_diff = ((LA.norm(S_new - S))**2)/4
#       percent_S_diff = 100*S_diff/S_new.size
        percent_cost_diff = 100*(cost_list[i] - cost_oldM)/cost_list[i]
        #percent_cost_diff = 100*(cost_list[i] - cost_list[i+1])/cost_list[i]
#       true_cost = setup_cost(projective_distance_matrix(Y),S)
#       true_cost_list.append(true_cost(Y_new.T))
        # Do an SVD to get the correlation matrix on the sphere.
        # Y,s,vh = LA.svd(out_matrix,full_matrices=False)
        D_fs = fubinistudy(complexify(Y_new.T))
        Z = np.where(D_fs < 1e-6)
        if not np.allclose(Z[0],Z[1]):
            print('WARNING: There are nearly colinear points (D_FS < 1e-6).')
        else:
            print('No colinear points found.')
        if verbosity > 0:
            print('Through %i iterations:' %(i+1))
#           print('\tTrue cost: %2.2f' %true_cost(Y_new.T))
            print('\tComputed cost: %2.2f' %cost_oldM)
            print('\tPercent cost difference: % 2.2f' %percent_cost_diff)
#           print('\tPercent Difference in S: % 2.2f' %percent_S_diff)
            print('\tComputed cost with new M: %2.2f' %cost_newM)
            if np.isnan(cost_newM):
                stuff = {'Y_new': Y_new, 'Y_old': Y,
                    'S_new': S_new, 'S_old': S, 'M_new': M_new, 'M_old': M,
                    'cost_fn_new': cost_new, 'cost_fn_old': cost, 'grad': egrad}
                return Y_new, stuff
#           print('\tDifference in cost matrix: %2.2f' %(LA.norm(C-C_new)))
#       if S_diff < 1:
#           print('No change in S matrix. Stopping iterations')
#           break
        if verbosity > 2:
            print(Y_new.T)
#       if percent_cost_diff < .0001:
#           print('No significant cost improvement. Stopping iterations.')
#           break
        if i == max_iter:
            print('Maximum iterations reached.')
        # Update variables:
        Y = Y_new
#       C = C_new
#       S = S_new
        M = M_new
    return Y, cost_list

def distance_to_weights(D):
    """Compute the weight matrix W from the distance matrix D.

    The weight matrix corresponding to a distance matrix `D = [d_ij]` is
    given by `W = [w_ij]` with

        .. :math:`w_{ij} = \frac{1}{\sqrt{1 - cos^2(d_ij)}}`.

    Since this is undefined when `d_ij = 0`, so we set the diagonal
    entries of `W` to 1.    

    Parameters
    ----------
    D : ndarray (n,n)
        Distance matrix. Must be square and contain no off-diagonal zeros.
    
    Returns
    -------
    W : ndarray (n,n)
        Weights matrix.

    """
    # TODO: no longer identical to pmds version. This version should
    # always be used.
    W_inv = (1 - np.cos(D)**2)     
    W = np.sqrt((W_inv+np.eye(D.shape[0]))**-1)
    np.fill_diagonal(W,1)
    return W

def setup_sum_cost(omega,M,D,W,p,return_derivatives=False):
    """docstring"""
    def F(Y):
        return 0.5*(sum([
            LA.norm(M[i]*W*(Y.T@mp(omega,i)@Y - np.cos(D)))**2 for i in range(p)]
            ))
    def dF(Y):
        return 2*(sum([
            mp(omega,p-i)@Y@(M[i]*W**2*(Y.T@mp(omega,i)@Y - np.cos(D)))
            for i in range(p)]
            ))
    if return_derivatives:
        return F, dF
    else:
        return F


###############################################################################
# Lens space utilities
###############################################################################

def real_ip(u,v):
    """Real inner product of complex vectors."""
    return np.real(u)@np.real(v) + np.imag(u)@np.imag(v)

def complex_as_matrix(z,n):
    """Represent a complex number as a matrix.
    
    Parameters
    ----------
    z : complex float
    n : int (even)
    
    Returns
    -------
    Z : ndarray (n,n)
        Real-valued n*n tri-diagonal matrix representing z in the ring of n*n matrices.
        
    """
    
    Z = np.zeros((n,n))
    ld = np.zeros(n-1)
    ld[0::2] = np.imag(z)
    np.fill_diagonal(Z[1:], ld)
    Z = Z - Z.T
    np.fill_diagonal(Z, np.real(z))
    return Z

def g_action_matrix(p,d):
    """Create a matrix corresponding to the action of Z_p on S^d. 
        
    Parameters
    ----------
    p : int
        Root of unity to use for quotient, i.e. p = 5 corresponds to
        5th roots of unity.
    d : int
        Dimension of sphere. Must be odd.

    Returns
    -------
    omega : ndarray
        Matrix defining the group action of Z_p on S^d. Always a
        block-diagonal (trilinear), orthogonal matrix.
    
    """

    # TODO: change this so that dimension is of ambient space.
    if d%2 == 0:
        raise ValueError('Sphere must be odd dimensional.')
    theta = 2*np.pi/p
    rot_block = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    omega = np.zeros((d+1,d+1))
    for i in range(0,d+1,2):
        omega[i:i+2,i:i+2] = rot_block
    return omega

def lens_distance_matrix(Y,rotations,p):
    """Find the true lens space distance matrix for a data.

    Parameters
    ----------
    Y : ndarray (k*n)
        Data on lens space with each column a data point. Complex valued
        with unit-norm columns.
    rotations : ndarray (n*n)
        For each pair of data points, `(y_i,y_j)` an integer `s` between
        `0` and `p` such that the arccosine of the complex inner product
        :math:`\langle y_i, \omega^p y_j \rangle` gives the correct
        distance.
    p : int
        Root of unity to use in computing distances.

    Returns
    -------
    D : ndarray (n*n)
        Distance matrix.

    """

    M = get_masks(rotations)
    omega = np.exp(2j*np.pi/p)
    D = sum(M[i]*(Y.T@(omega**i)*Y) for i in range(p))
    return D
 
def optimal_rotation_new(Y,p):
    """Choose the correct representative from each equivalence class.

    Parameters
    ----------
    Y : ndarray (d*n)
        Data, with one COLUMN for each of the n data points in
        d-dimensional space.
    w : ndarray (d*d)
        Rotation matrix for p-th root of unity.
    p : int
        Order of cyclic group.

    Returns
    -------
    S : ndarray (n*n)
        Correct power of w to use for each inner product. Satisfies
            S + S.T = 0 (mod p)

    """

    # (I'm not convinced this method is actually better, algorithmically.)
    # Convert Y to complex form:
    Ycplx = Y[0::2] + 1j*Y[1::2]
    cplx_ip = Ycplx.T@Ycplx.conjugate()
    ip_angles = np.angle(cplx_ip)
    ip_angles[ip_angles<0] += 2*np.pi   #np.angles uses range -pi,pi
    root_angles = np.linspace(0,2*np.pi,p+1)
    S = np.zeros(ip_angles.shape)
    for i in range(ip_angles.shape[0]):
        for j in range(ip_angles.shape[1]):
            S[i,j] = np.argmin(np.abs(ip_angles[i,j] - root_angles))
    S[S==p] = 0
    S = S.T     # Want the angle to act on the second component.
    return S

def optimal_rotation(Y,omega,p):
    maxYY = Y.T@Y
    S = np.zeros(np.shape(Y.T@Y))
    for i in range(1,p):
        tmpYY = Y.T@mp(omega,i)@Y
        S[tmpYY>maxYY] = i
        maxYY = np.maximum(maxYY,tmpYY)
    return S

def get_masks(S,p):
    """Turn matrix of correct powers into masks.

    Parameters
    ----------
    S : ndarray (n,n)
    p : int

    Returns
    M : list of ndarrays p*(n,n)

    """
    
    M = []
    for i in range(p):
        M.append(S==i)
    return M

def complexify(Y):
    """Convert data in 2k-dimensional real space to k-dimensional
    complex space.

    Parameters
    ----------
    Y : ndarray (2k,n)
        Real-valued array of data. Number of rows must be even.

    Returns
    -------
    Ycplx : ndarray (k,n)
        Complex-valued array of data.

    """

    Ycplx = Y[0::2] + 1j*Y[1::2]
    return Ycplx

def realify(Y):
    """Convert data in k-dimensional complex space to 2k-dimensional
    real space.

    Parameters
    ----------
    Y : ndarray (k,n)
        Real-valued array of data, `k` must be even.

    Returns
    -------
    Yreal : ndarray (2k,n)
        Complex-valued array of data.

    """

    if Y.ndim == 1:
        Yreal = np.zeros(2*Y.shape[0])
    else:
        Yreal = np.zeros((2*Y.shape[0],Y.shape[1]))
    Yreal[0::2] = np.real(Y)
    Yreal[1::2] = np.imag(Y)
    return Yreal

###############################################################################
# Highly Experimental Stuff
###############################################################################

def setup_fubini_study_cost(D,W):
    def F(Y):
        return (0.5*np.linalg.norm((Y.conj().T @ Y) * ((Y.conj().T @ Y).conj().T)
            -  np.cos(D)**2)**2)
    return F

def fubinistudy(X):
    """Distance matrix of X using Fubini-Study metric.
    
    Parameters
    ----------
    X : ndarray (complex, d,n)
        Data.
    Returns
    -------
    D : ndarray (real, n,n)
        Distance matrix.
    
    """
    
    D = np.arccos(np.sqrt((X.conj().T@X)*(X.conj().T@X).conj().T))
    np.fill_diagonal(D,0) # Things work better if diagonal is exactly zero.
    return np.real(D)

def FS_mds(
    Y,
    D,
    p,
    max_iter = 20,
    verbosity = 1,
    pmo_solve = 'cg',
    autograd = False,
    appx = False,
    minstepsize=1e-10,
    mingradnorm=1e-6
):
    """VERY EXPERIMENTAL.

    Attempts to align a collection of data points in the lens space
    :math:`L_p^n` so that the collection of distances between each pair
    of points matches a given input distance matrix as closely as
    possible.

    Parameters
    ----------
    Y : ndarray (m*d)
        Initial guess of data points. Each row corresponds to a point on
        the (2n-1)-sphere, so must have norm one and d must be even.
    D : ndarray (square)
        Distance matrix to optimize toward.
    p : int
        Cyclic group with which to act.
    max_iter: int, optional
        Maximum number of times to iterate the loop.
    verbosity: int, optional
        Amount of output to display at each iteration.
    pmo_solve: string, {'cg','sd','tr','nm','ps'}
        Solver to use with pymanopt. Default is conjugate gradient.

    Returns
    -------
    X : ndarray
        Optimal configuration of points on lens space.
    C : list (float)
        Computed cost at each loop of the iteration.
    T : list (float)
        Actual cost at each loop of the iteration.

    Notes
    -----
    The optimization can only be carried out w/r/t/ an approximation of
    the true cost function (which is not differentiable). The computed
    cost C should not match T, but should decrease when T does.

    """

    m = Y.shape[0]
    d = Y.shape[1] - 1
    W = distance_to_weights(D)
    omega = g_action_matrix(p,d)
    if appx:
        M = get_blurred_masks(Y.T,omega,p,D)
    else:
        S = optimal_rotation(Y.T,omega,p)
        M = get_masks(S,p)
    C = np.cos(D)
#   # TODO: verify that input is valid.
    cost = setup_sum_cost(omega,M,D,W,p)
    cost_list = [cost(Y.T)]
    manifold = Oblique(d+1,m) # Short, wide matrices.
    if pmo_solve == 'cg':
        solver = ConjugateGradient(
                minstepsize=minstepsize,
                mingradnorm=mingradnorm)
    elif pmo_solve == 'nm':
        solver = NelderMead()
    for i in range(0,max_iter):
        if autograd:
            cost = setup_sum_cost(omega,M,D,W,p)
            problem = pymanopt.Problem(manifold, cost, verbosity=verbosity)        
        else:
            cost, egrad = setup_sum_cost(omega,M,D,W,p,return_derivatives=True)
            problem = pymanopt.Problem(
                manifold,
                cost,
                egrad = egrad,
                verbosity = verbosity
            )
        if pmo_solve == 'cg' or pmo_solve == 'sd' or pmo_solve == 'tr':
            Y_new = solver.solve(problem, x=Y.T)
        else:
            Y_new =  solver.solve(problem)
        Y_new = Y_new.T     # Y should be tall-skinny
        cost_oldM = cost(Y_new.T)
        # cost_list.append(cost_oldM)
        S_new = optimal_rotation(Y_new.T,omega,p)
        M_new = get_masks(S_new,p)
        cost_new = setup_sum_cost(omega,M_new,D,W,p)
        cost_newM = cost_new(Y_new.T)
        cost_list.append(cost_newM)
        percent_cost_diff = 100*(cost_list[i] - cost_oldM)/cost_list[i]
        # NOW DO AN UPDATE RUN WITH FS-METRIC!
        print('Entering FS run.')
        fs_cost = setup_fubini_study_cost(D,W)
        problem = pymanopt.Problem(manifold,fs_cost,verbosity=verbosity)
        Ycplx = complexify(Y_new.T)
        Y_FS = solver.solve(problem, x = Ycplx)
        Y_FS = realify(Y_FS)
        cost_newFS = cost_new(Y_FS)
        if verbosity > 0:
            print('Through %i iterations:' %(i+1))
#           print('\tTrue cost: %2.2f' %true_cost(Y_new.T))
            print('\tComputed cost: %2.2f' %cost_oldM)
            print('\tPercent cost difference: % 2.2f' %percent_cost_diff)
#           print('\tPercent Difference in S: % 2.2f' %percent_S_diff)
            print('\tComputed cost with new M: %2.2f' %cost_newM)
            print('\tComputed cost after FS update: %2.2f' %cost_newFS)
            if np.isnan(cost_newM):
                stuff = {'Y_new': Y_new, 'Y_old': Y,
                    'S_new': S_new, 'S_old': S, 'M_new': M_new, 'M_old': M,
                    'cost_fn_new': cost_new, 'cost_fn_old': cost, 'grad': egrad}
                return Y_new, stuff
#           print('\tDifference in cost matrix: %2.2f' %(LA.norm(C-C_new)))
#       if S_diff < 1:
#           print('No change in S matrix. Stopping iterations')
#           break
        if percent_cost_diff < .0001:
            print('No significant cost improvement. Stopping iterations.')
            break
        if i == max_iter:
            print('Maximum iterations reached.')
        # Update variables:
        Y = Y_FS
#       C = C_new
#       S = S_new
        M = M_new
    return Y, cost_list

