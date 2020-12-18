"""Lens Space MDS based on Universal MDS. """
##############
# Imports #{{{
##############

import numpy as np
from numpy import linalg as LA

#####################################}}}
# GEOMETRIC METHODS FOR LENS SPACES #{{{
########################################

# Define a global variable for cyclic group order outside of all
# functions.
p = 3

def real_ip(u, v):
    """Real inner product of complex vectors."""
    ip = np.real(np.vdot(u,v))
    return ip

def dist_rep(x, y, p=3):
    """Give the correct representative of the lens equivalence class and
    the distance between classes.
    
    In lens space, `y` is equivalent to `alpha*y` for `alpha` in the
    cyclic group Z_p.  The correct representative of the equivalence
    class of `y` with respect to `x` is the one which maximizes the real
    inner product between them (i.e. minimizes the angle), and this
    minimum angle is the distance between points.
    
    Parameters
    ----------
    x : complex ndarray (n,)
        Vector representing equivalence class of base point.
    y : complex ndarray (n,)
        Vector representing equivalence class of another point.
    p : int
        Cyclic group to use.
        
    Returns
    -------
    dist : float
        Distance between `x` and `y`.
    y_opt : complex ndarray (n,)
        Element of equivalence class of `m` closets to `x`.
    k : int
        Power of rotation necessary to give y_opt.
        
    """
    # TODO: the return object here is a little ugly. Consider
    # refactoring this function into the component pieces (especially if
    # doing so could reduce computation).   

    opt_set = [real_ip(x,np.exp(2j*np.pi*alpha/p)*y) for alpha in range(0,p)]
    k = np.argmax(opt_set)
    dist = np.arccos(min([1,opt_set[k]]))
    y_opt = np.exp(2j*np.pi*k/p)*y
    return dist, y_opt, k

def lens_exp(x, v, p=3):
    """Exponential map on lens space.
    
    Parameters
    ----------
    x : ndarray (n,)
        Point on sphere.
    v : ndarray (n,)
        Tangent vector to `x`. Must satisfy real_ip(x,v) = 0.
    p : int
        Cyclic group order.
    
    Returns
    -------
    y : ndarray (n,)
    
    """
    
    if np.allclose(v, np.zeros(v.shape)):
        y_opt = x
        return y_opt
    if np.abs(real_ip(x,v)) > 1e-4:
        print('WARNING: computing exp with vector which may not be tangent '\
            'to x. Inner product was %2.5f' %real_ip(x,v))
        # raise ValueError('Vector v not tangent to unit vector x.')
    y = np.cos(LA.norm(v))*x + np.sin(LA.norm(v))*v/LA.norm(v)
    _, y_opt, _ = dist_rep(x, y, p=p)
    return y_opt
    
def lens_log(x, y, p=3, normalize=False):
    """Inverse exponential map on lens space."""
    
    # TODO: debugging. Require input to be unit norm.
    assert(np.allclose(LA.norm(x), 1.0))
    assert(np.allclose(LA.norm(y), 1.0))
    dist, y_opt, _ = dist_rep(x, y)
    proj = y_opt - real_ip(y_opt,x)*x
    if normalize:
        dist = 1.0
    if LA.norm(proj) > 1e-8:
        v = dist*proj/LA.norm(proj)
    else:
        v = np.zeros(x.shape, dtype=complex)
    if np.abs(real_ip(v,x)) > 1e-5:
        print('WARNING: log vector may be non-orthogonal. Inner product '\
            'was %2.5f' %(real_ip(v,x)))
    return v

def lens_dist(x, y, p=3):
    d = dist_rep(x, y, p=p)[0]
    return d

def nearest_point(x, y, r, p=3, tol=1e-6):
    """Construct the nearest point to y on the circle of radius r at x."""
    
    y_hat = lens_exp(x, r*lens_log(x, y, p=p, normalize=True), p=p)
    d,_,_ = dist_rep(x, y_hat, p=p)
    if np.abs(d - r) > tol:
        y_hat = cpn_extend(x, y, r)
    # v = lens_log(proj, y, p=p)
    #t_up = 1.0
    #t_down = 0.0
    #count = 0
    # Binary search along geodesic from proj to y until reach distance r.
    #while np.abs(d - r) > tol:
    #    print('Loop number %d' %count)
    #    t = (t_up - t_down) / 2
    #    y_hat = lens_exp(proj, t*v, p=p)
    #    d,_,_ = dist_rep(x, y_hat, p=p)
    #    if d - r > 0:
    #        t_down = t
    #    else:
    #        t_up = t
    #    count += 1
    #    if count > 30:
    #        break
    return y_hat

####################################}}}
# COMPLEX PROJECTIVE SPACE METHODS #{{{
#######################################

def cpn_extend(x, y, r):
    """Extend along a geodesic in complex projective space.
    
    Parameters
    ----------
    x : complex ndarray (n,)
        Starting point for geodesic.
    y : complex ndarray (n,)
        Ending point for geodesic.
    r : float
        Length of geodesic ray. Must be positive.
        
    """
    
    n = x.shape[0]
    M = (np.eye(n) - np.outer(x,x.conj()))@y*((np.vdot(x,y))**-1)
    U, s, Vh = LA.svd(np.reshape(M,(n,1)),full_matrices=False)
    U = U.flatten()
    theta = np.arctan(s)
    t = r / fubini_study(x,y)
    Gt = x*np.cos(theta*t) + U*np.sin(theta*t)
    return Gt

def fubini_study(x,y):
    """Compute the Fubini-Study distance between vectors.
    
    Parameters
    ----------
    x : complex ndarray (n,)
    y : complex ndarray (n,)
    
    Returns
    -------
    d : float
        Fubini-Study distance.
        
    """
    
    prod = np.real(np.vdot(x, y)*np.vdot(y, x))
    prod = max(np.sqrt(prod), 0.0)
    prod = min(prod, 1.0)
    d = np.arccos(prod)
    return d

###########################}}}
# MDS COMPONENT FUNCTIONS #{{{
##############################

def lens_weiszfeld(
    X,
    m,
    W=None,
    p=3,
    alpha=1.5,
    mincostdiff=1e-10,
    mingradnorm=1e-10,
    maxiters=500,
    verbosity=0
):
    """Weiszfeld algorithm on lens space.
    
    Parameters
    ----------
    W : ndarray (n,)
        Weights.
    X : complex ndarray (k,n)
        Point set on lens space as `n` column vectors. Columns must have unit norm.
    m : complex ndarray (k,)
        Initial point.
    alpha : float
        Step size. To guarantee convergence needs to be in [1,2].
    mincostdiff : float, optional
        Convergence parameter for cost improvement.
    mingradnorm : float, optional
        Convergence parameter for gradient step size.
    maxiters : int, optional
        Maximum iterations permitted. Default is 500. Mainly serves to
        prevent infinite loops -- convergence should be achieved in far
        fewer iterations.
    verbosity : int, optional
        Amount of information to print to stdout. `0` suppresses all
        output, `1` is useful for testing performance, and higher for
        debugging.
        
    Returns
    -------
    m : complex ndarray (k,)
        Median of points `X`.
        
    """
    
    assert X.ndim == 2
    n = X.shape[1]
    k = X.shape[0]
    if W is None:
        W =  np.ones(n)
    count = 0
    cost_old = np.inf
    m_old = m
    m_list = np.zeros((k,1),dtype=complex)
    m_list[:,0] = m_old
    if verbosity > 0:
        print('%%% Starting Weiszfeld Algorithm. %%%')
    while True:
        if verbosity > 0:
            print('Iteration %d' %count)
        beta = 0
        cost = 0
        at_vertex = -1
        v = np.zeros(k)
        for i in range(n):
            d = lens_dist(m, X[:,i])
            # TODO: what is the right value for this tolerance?
            if d > 1e-8:
                v = v + (W[i]/d)*lens_log(m, X[:,i], p=p)
                beta += W[i]/d
                cost += W[i]*d
            else:
                if verbosity > 1:
                    print('Found that m is equal to a vertex in X.')
                at_vertex = i
        # TODO: does this case need to be handled better? I think Ostresh is
        # actually doing the following, which doesn't quite match Fletcher.
        gradnorm = LA.norm(v)
        if at_vertex >= 0:
            if gradnorm <= W[at_vertex]:
                v = np.zeros(v.shape, dtype=complex)
                if beta == 0.0:
                    beta = 1.0
                    # Setting beta=0 handles the weird edge case where
                    # all vertices are the same point, and m has also
                    # converged to that point.
                if verbosity > 1:
                    print('Setting v to zeros.')
                # TODO: shouldn't this really just mean stop here?
            else:
                v = v - W[at_vertex]*v/gradnorm
        if verbosity > 0:
            print('\tCost after iteration %d: %2.6f' %(count,cost))
            print('\tCost difference: %2.6f' %(cost_old - cost))
        if verbosity > 1:
            with np.printoptions(precision=3):
                print('\tPseudo gradient vector:', v)
            print('\tNorm of pseudo gradient: %2.6f' %(gradnorm))
            print('\tCheck v still orthogonal to m: %2.6f' %(real_ip(v,m)))
            print('\tbeta: %2.6f' %beta)
        if cost_old - cost < mincostdiff:
            if verbosity > 0:
                print('Cost improvement below threshold. Stopping.')
            # m = m_old
            # Weiszfeld is contractive and in theory should never
            # "overshoot" so in theory it is fine to return m even if
            # the cost improvement was small. It will still be an
            # improvement.
            break
        if gradnorm < mingradnorm:
            if verbosity > 0:
                print('Pseudo gradient norm below threshold. Stopping.')
            break
        m_old = m
        m = lens_exp(m, alpha*(beta**-1)*v, p=p)
        m_list = np.hstack((m_list, np.reshape(m,(2,1))))
        cost_old = cost
        count += 1
        if count >= maxiters:
            if verbosity > 0:
                print('Reached maximum allowed iterations (%d).' %maxiters)
            break
    return m

def lens_place(y,
    X,
    Dy,
    mincostdiff=1e-10,
    p=3,
    verbosity=0
):
    """Place routine for lens space
    
    Parameters
    ----------
    y : ndarray (k,)
        Point to update.
    X : ndarray (k,n)
        Set of data points.
    Dy : ndarray (n,)
        Distances from `y` to each `X.`
    mincostdiff : float, optional
        Threshold for convergence.
    p : int, optional
        Cyclic group. Default is Z_3.
    
    Returns
    -------
    y_opt ndarray (k,)
        Updated `y` which minimizes sum of `|X - Dy|`.
    
    """
    
    X_hat = np.zeros(X.shape, dtype=complex)
    n = Dy.shape[0]
    epsilon = component_cost(y, X, Dy)
    count = 0
    if verbosity > 0:
        print('%%% Starting PLACE Algorithm. %%%')
        print('Initial cost: %2.6f' %epsilon)
    while True:
        count += 1
        print('Begun place loop %d' %count)
        for j in range(n):
            X_hat[:,j] = nearest_point(X[:,j], y, Dy[j], p=p)
        # TODO: is this including y as a vertex in the Weiszfeld
        # iteration? If so, probably don't want that. Weiszfeld
        # correctly handles starting at a vertex, but since distance to
        # self is always zero it should have weight zero in the update.
        y = lens_weiszfeld(X_hat, y, p=p, verbosity=verbosity-1)
        new_epsilon = component_cost(y, X, Dy)
        if verbosity > 0:
            print('Cost after PLACE iteration %d: %2.4f' %(count,new_epsilon))
            print('Difference in cost: %2.4f' %(epsilon - new_epsilon))
        if epsilon - new_epsilon < mincostdiff:
            break
        else:
            epsilon = new_epsilon
    return y

def component_cost(y, X, Dy, p=3):
    """Component of cost function for single element."""
    C = 0
    for j in range(Dy.shape[0]):
        C += np.abs(lens_dist(y, X[:,j], p=p) - Dy[j])
    return C

def lens_dist_mtx(X, Y, p=3):
    """Lens distance matrix."""
    
    assert X.shape[0] == Y.shape[0]
    
    if X.ndim == 1:
        m = 1
        X = np.reshape(X, (X.shape[0],1))
    else:
        m = X.shape[1]
    if Y.ndim == 1:
        n = 1
        Y = np.reshape(Y, (Y.shape[0],1))
    else:
        n = Y.shape[1]
    D = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            D[i,j] = lens_dist(X[:,i], Y[:,j], p=p)
    # TODO: can speed this up by not recomputing all the duplicate entries in the lower half of the array, but only when computing d(X,X).
    #if np.array_equal(X, Y):
    #    # If the input is distance from X to itself, return a true diagonal distance matrix.
    #    D = D + D.T
    if m == 1 or n == 1:
        D = D.flatten()
    return D

def lens_PlaceCenter(D, X0, p=3, tol=.001, verbosity=0):
    """PlaceCenter routine for lens spaces.
    
    Parameters
    ----------
    D : ndarray (n,n)
        Goal distance matrix.
    X0 : ndarray (k,n)
        Some initial guess.
    p : int, optional
        Cyclic group order. Defualt is `p = 3`.
        
    Returns
    -------
    X : ndarray (k,n)
        Optimized point cloud.
        
    """
    
    X = X0.copy()
    count = 0
    print('Starting PlaceCenter')
    while True:
        count += 1
        print('PlaceCenter loop %d' %count)
        epsilon = full_cost(X, D, p=p)
        for i in range(D.shape[0]):
            X[:,i] = lens_place(X[:,i], X, D[i,:], p=p, verbosity=verbosity)
        new_epsilon = full_cost(X, D, p=p)
        if epsilon - new_epsilon < tol:
            break
    print('Total loops: %d' %count)
    return X

def full_cost(X, D, p=3):
    """Doing this the dumb way for now."""
    n = D.shape[0]
    C = 0
    for i in range(n):
        for j in range(n):
            C += np.abs(lens_dist(X[:,i], X[:,j], p=p) - D[i,j])
    return C        

#############################}}}
