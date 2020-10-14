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


def do_tests(Y,D,verbose=0):
    X,C,T1 = projective_mds.pmds(Y,D,weighted=False,verbose=verbose)
    fig = plt.figure(1,constrained_layout=True)
    gs = fig.add_gridspec(6,5)
    ax = fig.add_subplot(gs[1:6,:],projection='3d')
    ax = projective_mds.plot_RP2(X,ax)
    cost_ax = fig.add_subplot(gs[0,:])
    cost_ax.plot(C)
    plt.suptitle('Unweighted Frobenius')

    X,C,T2 = projective_mds.pmds(Y,D,weighted=True,verbose=verbose)
    fig = plt.figure(2,constrained_layout=True)
    gs = fig.add_gridspec(6,5)
    ax = fig.add_subplot(gs[1:6,:],projection='3d')
    ax = projective_mds.plot_RP2(X,ax)
    cost_ax = fig.add_subplot(gs[0,:])
    cost_ax.plot(C)
    plt.suptitle('Weighted Frobenius')

    X,C,T3 = projective_mds.pmds(Y,D,solve_prog='autograd',appx='taylor',verbose=verbose)
    fig = plt.figure(3,constrained_layout=True)
    gs = fig.add_gridspec(6,5)
    ax = fig.add_subplot(gs[1:6,:],projection='3d')
    ax = projective_mds.plot_RP2(X,ax)
    cost_ax = fig.add_subplot(gs[0,:])
    cost_ax.plot(C)
    plt.suptitle('Taylor Series')

    X,C,T4 = projective_mds.pmds(Y,D,solve_prog='autograd',appx='rational',verbose=verbose)
    fig = plt.figure(4,constrained_layout=True)
    gs = fig.add_gridspec(6,5)
    ax = fig.add_subplot(gs[1:6,:],projection='3d')
    ax = projective_mds.plot_RP2(X,ax)
    cost_ax = fig.add_subplot(gs[0,:])
    cost_ax.plot(C)
    plt.suptitle('Rational Approx')

    X,C,T5 = projective_mds.pmds(Y,D,solve_prog='autograd',appx='frobenius',verbose=verbose)
    fig = plt.figure(5,constrained_layout=True)
    gs = fig.add_gridspec(6,5)
    ax = fig.add_subplot(gs[1:6,:],projection='3d')
    ax = projective_mds.plot_RP2(X,ax)
    cost_ax = fig.add_subplot(gs[0,:])
    cost_ax.plot(C)
    plt.suptitle('Weighted Frobenius (autograd)')
    plt.show()
    return T1[-1],T2[-1],T3[-1],T4[-1],T5[-1]




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
