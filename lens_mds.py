""" Dimensionality reduction on Lens spaces using an MDS-type method. """
import autograd.numpy as np
import autograd.numpy.linalg as LA
from autograd.numpy.linalg import matrix_power as mp
import matplotlib.pyplot as plt
import pymanopt
from pymanopt.solvers import *
from pymanopt.manifolds import Oblique

def lmds(Y,D,p,max_iter=20,verbosity=1,pmo_solve='cg',autograd=True,appx=False):
    """Lens multi-dimensional scaling algorithm.

    Parameters
    ----------
    Y : ndarray (m*d)
        Initial guess of data points. Each row corresponds to a point on
        the (2n-1)-sphere, so must have norm one and d must be even.
    D : ndarray (square)
        Distance matrix to optimize toward.
    p : int
        Cyclic group with which to act. Must be prime (this is not
        checked).
    max_iter: int, optional
        Number of times to iterate the loop.
    verbosity: int, optional
        Amount of output to display at each iteration.
    pmo_solve: string, {'cg','sd','tr','nm','ps'}
        Solver to use with pymanopt. Default is conjugate gradient.

    Returns
    -------
    X : ndarray
        Optimal location of data points.
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
#   if d%2 == 0:
#       raise ValueError('Input data matrix must have an even number of ' +
#           'columns (must be on an odd-dimensional sphere). Given data had ' +
#           '%i columns.',%(d+1))
    # TODO: verify that input is valid.
    cost = setup_sum_cost(omega,M,D,W,p)
    cost_list = [cost(Y.T)]
#   true_cost = setup_cost(projective_distance_matrix(Y),S)
#   true_cost = setup_cost(Y,omega,S,D,W)
#   true_cost_list = [true_cost(Y.T)]
    manifold = Oblique(d+1,m) # Short, wide matrices.
    if pmo_solve == 'cg':
        solver = ConjugateGradient()
    elif pmo_solve == 'nm':
        solver = NelderMead()
    # TODO: implement and experiment with other solvers.
    for i in range(0,max_iter):
        if autograd:
            cost = setup_sum_cost(omega,M,D,W,p)
            problem = pymanopt.Problem(manifold, cost, verbosity=verbosity)        
        else:
            cost, egrad = setup_sum_cost(omega,M,D,W,p,return_derivatives=True)
            problem = pymanopt.Problem(manifold, cost, egrad=egrad, verbosity=verbosity)
        if pmo_solve == 'cg' or pmo_solve == 'sd' or pmo_solve == 'tr':
            Y_new = solver.solve(problem,x=Y.T)
        else:
            Y_new =  solver.solve(problem)
        Y_new = Y_new.T     # Y should be tall-skinny
        cost_oldM = cost(Y_new.T)
        cost_list.append(cost_oldM)
        if appx:
            M_new = get_blurred_masks(Y_new.T,omega,p,D)
        else:
            S_new = optimal_rotation(Y_new.T,omega,p)
            M_new = get_masks(S_new,p)
        cost_new = setup_sum_cost(omega,M_new,D,W,p)
        cost_newM = cost_new(Y_new.T)
#       S_diff = ((LA.norm(S_new - S))**2)/4
#       percent_S_diff = 100*S_diff/S_new.size
        percent_cost_diff = 100*(cost_list[i] - cost_list[i+1])/cost_list[i]
#       true_cost = setup_cost(projective_distance_matrix(Y),S)
#       true_cost_list.append(true_cost(Y_new.T))
        # Do an SVD to get the correlation matrix on the sphere.
        # Y,s,vh = LA.svd(out_matrix,full_matrices=False)
        if verbosity > 0:
            print('Through %i iterations:' %(i+1))
#           print('\tTrue cost: %2.2f' %true_cost(Y_new.T))
            print('\tComputed cost: %2.2f' %cost_list[i+1])
            print('\tPercent cost difference: % 2.2f' %percent_cost_diff)
#           print('\tPercent Difference in S: % 2.2f' %percent_S_diff)
            print('\tComputed cost with new M: %2.2f' %cost_newM)
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
        Y = Y_new
#       C = C_new
#       S = S_new
        M = M_new
    return Y, cost_list

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
    if d%2 == 0:
        raise ValueError('Sphere must be odd dimensional.')
    theta = 2*np.pi/p
    rot_block = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    omega = np.zeros((d+1,d+1))
    for i in range(0,d+1,2):
        omega[i:i+2,i:i+2] = rot_block
    return omega

def distance_to_weights(D):
    """Compute the weight matrix W from the distance matrix D."""
    # TODO: currently identical to pmds version. Remark if changes.
    W_inv = (1 - np.cos(D)**2)     
    W = np.sqrt((W_inv+np.eye(D.shape[0],D.shape[1]))**-1 - np.eye(D.shape[0],D.shape[1]))
    return W

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

def acos_validate(M):
    """Replace values in M outside of domain of acos with +/- 1."""
    big_vals = M >= 1.0
    M[big_vals] = 1.0
    small_vals = M <= -1.0
    M[small_vals] = -1.0
    return M

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

def setup_sum_cost(omega,M,D,W,p,return_derivatives=False):
    """docstring"""
    def F(Y):
        return 0.5*(sum([
            LA.norm(M[i]*W*(Y.T@mp(omega,i)@Y) -
            M[i]*W*np.cos(D))**2 for i in range(p)]
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

def get_blurred_masks(Y,omega,p,D):
    """Get approximate versions of the masks by integrating."""

    M = []
    tmp = np.zeros(np.size(Y.T@Y))
    for i in range(p):
        tmp = 1/np.abs(D - np.arccos(acos_validate(Y.T@mp(omega,i)@Y)))
#       tmp = 1/np.arccos(acos_validate(Y.T@mp(omega,i)@Y))
        M.append(tmp)
    M = np.nan_to_num(M/sum(M),nan=1.0)
    return M

