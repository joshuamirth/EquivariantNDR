import numpy as np
import scipy.io
import matlab.engine

# Example from Grubisic & Pietersz, "Rank Reduction Correlation Matrices", p2.
# Target matrix of high rank:
C = np.array([[1.0000,0.6124,0.6124],[0.6124,1.0000,0.8333],[0.6124,0.8333,1.0000]])
# Initial guess:
Y0 = np.array([[1.0000,0],[0.7112,0.7030],[0.6605,0.7508]])
# Weight matrix:
W = np.ones((3,3))
# Known solution from paper:
Yn = np.array([[1.0000,0],[0.6124,0.7906],[0.6124,0.7906]])
# Goal rank
d = 2
# Save to .mat as a crude way to pass to matlab.
scipy.io.savemat('ml_tmp.mat', dict(C=C,W=W,Y0=Y0,d=d))
# Run lrcm_min in matlab.
print('Starting MATLAB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
eng = matlab.engine.start_matlab()
t = eng.lrcm_wrapper()
print('MATLAB complete %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
# Load result from matlab.
workspace = scipy.io.loadmat('py_tmp.mat')
out_matrix = workspace['optimal_matrix']
test = np.linalg.norm(out_matrix - Yn)
print('Difference from known result: ' + str(test))
