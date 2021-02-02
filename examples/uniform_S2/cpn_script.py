"""Script for running CPn example computations on HPCC."""
import numpy as np
import cplx_projective

maxiter = 10
stuff = np.load('cp_test.npz')
K = stuff['K']
D_uni = stuff['D_uni']
Xopt, Areal, Aimag = cplx_projective.cp_mds_reg(K, D_uni, lam=1.0, v=1, maxiter=maxiter)
np.savez('output.npz', Xopt=Xopt, Areal=Areal, Aimag=Aimag)
