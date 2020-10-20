import time
import projective_mds
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def test1():
    T,n = projective_mds.circleRPn()
    D = projective_mds.graph_distance_matrix(T,k=5)
    Y = projective_mds.initial_guess(T,2)
    out = do_tests(Y,D)
    return out

def test2():
    T,n = projective_mds.circleRPn(noise=True)
    D = projective_mds.graph_distance_matrix(T,k=8)
    Y = projective_mds.initial_guess(T,2)
    out = do_tests(Y,D)
    return out

def test3():
    Ws = np.load('workspace.npz')
    T = Ws['BB']
    Y = T
    D = projective_mds.graph_distance_matrix(T,k=8)
    out = do_tests(Y,D)
    return out    

def test4():
    Ws = np.load('workspace.npz')
    T = Ws['BB']
    Y = Ws['Y']
    D = projective_mds.graph_distance_matrix(T,k=8)
    out = do_tests(Y,D)
    return out 

def test5():
    T,_ = projective_mds.circleRPn(noise=True,segment_points=60)
    D = projective_mds.graph_distance_matrix(T,k=8)
    Y,_ = projective_mds.circleRPn(dim=2,segment_points=100,num_segments=2,noise=True,v=0.1)
    out = do_tests(Y,D)
    return out

def test6():
    """Test using the bezier curve on RP^4 (which gets tangled by PPCA)."""
    B = np.load('bez_test.npy')
    D = projective_mds.graph_distance_matrix(B,k=5)
    Y = projective_mds.initial_guess(B,2)
    out = do_tests(Y,D)
    return out

def test7():
    """Test applying PMDS before PPCA, with the tangled curve."""
    B = np.load('bez_test.npy')
    D = projective_mds.graph_distance_matrix(B,k=5)
    X,C = do_tests(B,D,plot=False)
    Y1 = projective_mds.initial_guess(X[0],2)
    Y2 = projective_mds.initial_guess(X[1],2)
    Y3 = projective_mds.initial_guess(X[2],2)
    Y4 = projective_mds.initial_guess(X[3],2)
    Y5 = projective_mds.initial_guess(X[4],2)
    out = (Y1,Y2,Y3,Y4,Y5)
    return out

def test8():
    """Same as test1, but now comparing different autograd methods."""
    T,n = projective_mds.circleRPn()
    D = projective_mds.graph_distance_matrix(T,k=5)
    Y = projective_mds.initial_guess(T,2)
    costs, times = do_autograd_tests(Y,D)
    return costs, times

def test9():
    """Same as test4, but with autograd methods."""
    Ws = np.load('workspace.npz')
    T = Ws['BB']
    Y = Ws['Y']
    D = projective_mds.graph_distance_matrix(T,k=8)
    costs, times = do_autograd_tests(Y,D)
    return costs, times

def test10(solve_prog='autograd'):
    """Test using the bezier curve on RP^4 (which gets tangled by PPCA)."""
    B = np.load('bez_test.npy')
    D = projective_mds.graph_distance_matrix(B,k=5)
    Y = projective_mds.initial_guess(B,2)
    out = do_autograd_tests(Y,D,solve_prog=solve_prog)
    return out

def test11(pmo_solve='cg'):
    B = np.load('bez_test.npy')
    D = projective_mds.graph_distance_matrix(B,k=5)
    Y = projective_mds.initial_guess(B,2)
    out = compare_gradients(Y,D,pmo_solve=pmo_solve)
    return out

def test12(pmo_solve='cg'):
    """Same as test4, but with autograd methods."""
    Ws = np.load('workspace.npz')
    T = Ws['BB']
    Y = Ws['Y']
    D = projective_mds.graph_distance_matrix(T,k=8)
    out = compare_gradients(Y,D,pmo_solve=pmo_solve)
    return out

def do_tests(Y,D,plot=True):
#   X1,C1,T1 = projective_mds.pmds(Y,D,weighted=False,verbose=verbose)
    X2,C2,T2 = projective_mds.pmds(Y,D)
#    X3,C3,T3 = projective_mds.pmds(Y,D,solve_prog='autograd',appx='taylor',verbose=verbose)
#    X4,C4,T4 = projective_mds.pmds(Y,D,solve_prog='autograd',appx='rational',verbose=verbose)
    X5,C5,T5 = projective_mds.pmds(Y,D,autograd=True)

    if plot:
#       fig = plt.figure(1,constrained_layout=True)
#       gs = fig.add_gridspec(6,5)
#       ax = fig.add_subplot(gs[1:6,:],projection='3d')
#       ax = projective_mds.plot_RP2(X1,ax)
#       cost_ax = fig.add_subplot(gs[0,:])
#       cost_ax.plot(C1)
#       plt.suptitle('Unweighted Frobenius')

        fig = plt.figure(2,constrained_layout=True)
        gs = fig.add_gridspec(6,5)
        ax = fig.add_subplot(gs[1:6,:],projection='3d')
        ax = projective_mds.plot_RP2(X2,ax)
        cost_ax = fig.add_subplot(gs[0,:])
        cost_ax.plot(C2)
        plt.suptitle('Weighted Frobenius')

#       fig = plt.figure(3,constrained_layout=True)
#       gs = fig.add_gridspec(6,5)
#       ax = fig.add_subplot(gs[1:6,:],projection='3d')
#       ax = projective_mds.plot_RP2(X3,ax)
#       cost_ax = fig.add_subplot(gs[0,:])
#       cost_ax.plot(C3)
#       plt.suptitle('Taylor Series')

#       fig = plt.figure(4,constrained_layout=True)
#       gs = fig.add_gridspec(6,5)
#       ax = fig.add_subplot(gs[1:6,:],projection='3d')
#       ax = projective_mds.plot_RP2(X4,ax)
#       cost_ax = fig.add_subplot(gs[0,:])
#       cost_ax.plot(C4)
#       plt.suptitle('Rational Approx')

        fig = plt.figure(5,constrained_layout=True)
        gs = fig.add_gridspec(6,5)
        ax = fig.add_subplot(gs[1:6,:],projection='3d')
        ax = projective_mds.plot_RP2(X5,ax)
        cost_ax = fig.add_subplot(gs[0,:])
        cost_ax.plot(C5)
        plt.suptitle('Weighted Frobenius (autograd)')
        plt.show()

    points = (X2,X5)
    costs = (T2[-1],T5[-1])
    return points, costs

def do_autograd_tests(Y,D,verbose=1,solve_prog='autograd'):
    tic = time.perf_counter()
    X,C,T1 = projective_mds.pmds(Y,D,solve_prog=solve_prog,appx='frobenius',verbose=verbose,pmo_solve='cg')
    toc = time.perf_counter()
    t1 = toc - tic
    fig = plt.figure(1,constrained_layout=True)
    gs = fig.add_gridspec(6,5)
    ax = fig.add_subplot(gs[1:6,:],projection='3d')
    ax = projective_mds.plot_RP2(X,axes=ax)
    cost_ax = fig.add_subplot(gs[0,:])
    cost_ax.plot(C)
    plt.suptitle('Weighted Frobenius (Conjugate Gradient)')

    tic = time.perf_counter()
    X,C,T2 = projective_mds.pmds(Y,D,solve_prog=solve_prog,appx='frobenius',verbose=verbose,pmo_solve='nm')
    toc = time.perf_counter()
    t2 = toc - tic
    fig = plt.figure(2,constrained_layout=True)
    gs = fig.add_gridspec(6,5)
    ax = fig.add_subplot(gs[1:6,:],projection='3d')
    ax = projective_mds.plot_RP2(X,axes=ax)
    cost_ax = fig.add_subplot(gs[0,:])
    cost_ax.plot(C)
    plt.suptitle('Weighted Frobenius (Nelder-Mead)')

    tic = time.perf_counter()
    X,C,T3 = projective_mds.pmds(Y,D,solve_prog=solve_prog,appx='frobenius',verbose=verbose,pmo_solve='sd')
    toc = time.perf_counter()
    t3 = toc - tic
    fig = plt.figure(3,constrained_layout=True)
    gs = fig.add_gridspec(6,5)
    ax = fig.add_subplot(gs[1:6,:],projection='3d')
    ax = projective_mds.plot_RP2(X,axes=ax)
    cost_ax = fig.add_subplot(gs[0,:])
    cost_ax.plot(C)
    plt.suptitle('Weighted Frobenius (Steepest Descent)')

    tic = time.perf_counter()
    X,C,T4 = projective_mds.pmds(Y,D,solve_prog=solve_prog,appx='frobenius',verbose=verbose,pmo_solve='tr')
    toc = time.perf_counter()
    t4 = toc - tic
    fig = plt.figure(4,constrained_layout=True)
    gs = fig.add_gridspec(6,5)
    ax = fig.add_subplot(gs[1:6,:],projection='3d')
    ax = projective_mds.plot_RP2(X,axes=ax)
    cost_ax = fig.add_subplot(gs[0,:])
    cost_ax.plot(C)
    plt.suptitle('Weighted Frobenius, (Trust Regions)')

    tic = time.perf_counter()
    X,C,T5 = projective_mds.pmds(Y,D,solve_prog=solve_prog,appx='frobenius',verbose=verbose,pmo_solve='ps')
    toc = time.perf_counter()
    t5 = toc - tic
    fig = plt.figure(5,constrained_layout=True)
    gs = fig.add_gridspec(6,5)
    ax = fig.add_subplot(gs[1:6,:],projection='3d')
    ax = projective_mds.plot_RP2(X,axes=ax)
    cost_ax = fig.add_subplot(gs[0,:])
    cost_ax.plot(C)
    plt.suptitle('Weighted Frobenius (Particle Swarm)')

    plt.show()
    points = (X1,X2,X3,X4,X5)
    costs = (T1[-1],T2[-1],T3[-1],T4[-1],T5[-1])
    times = (t1,t2,t3,t4,t5)
    return points, costs, times

def compare_gradients(Y,D,pmo_solve='cg'):
    """Run the same test on autograd and analytic gradient."""
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('Solving with analytic gradient.')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    X_anal,C_anal,T_anal = projective_mds.pmds(Y,D,solve_prog='pymanopt',verbose=2,pmo_solve=pmo_solve)   
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print('Solving with autograd.')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    X_auto,C_auto,T_auto = projective_mds.pmds(Y,D,solve_prog='autograd',appx='frobenius',verbose=2,pmo_solve=pmo_solve)
    return X_auto, X_anal

# Example from Grubisic & Pietersz, "Rank Reduction Correlation Matrices", p2.
# Target matrix of high rank:
#C = np.array([[1.0000,0.6124,0.6124],[0.6124,1.0000,0.8333],[0.6124,0.8333,1.0000]])
# Initial guess:
#Y0 = np.array([[1.0000,0],[0.7112,0.7030],[0.6605,0.7508]])
# Weight matrix:
#W = np.ones((3,3))
# Known solution from paper:
#Yn = np.array([[1.0000,0],[0.6124,0.7906],[0.6124,0.7906]])
# Goal rank
#d = 2
# Save to .mat as a crude way to pass to matlab.
#scipy.io.savemat('ml_tmp.mat', dict(C=C,W=W,Y0=Y0,d=d))
# Run lrcm_min in matlab.
#print('Starting MATLAB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
#eng = matlab.engine.start_matlab()
#t = eng.lrcm_wrapper()
#print('MATLAB complete %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
# Load result from matlab.
#workspace = scipy.io.loadmat('py_tmp.mat')
#out_matrix = workspace['optimal_matrix']
#test = np.linalg.norm(out_matrix - Yn)
#print('Difference from known result: ' + str(test))
