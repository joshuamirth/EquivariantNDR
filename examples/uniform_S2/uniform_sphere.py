import numpy as np

# Create data for a uniform sphere.
rng = np.random.default_rng()
N = 144
dim = 2
S = rng.standard_normal((dim+1, N))
S = S / np.linalg.norm(S, axis=0)
np.savez('uniform_sphere_N%i_dim%i.npz' %(N,dim), S=S)
