# Script for generating images of point.
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

# Setup parameters
N = 12  # Create N*N images
amp = 1
sd = .8 # Standard deviation needs to be large enough for overlap.
K = 16  # Create K^2 total images.

# Generate the images:
x = np.linspace(-1,1,N)
y = np.linspace(-1,1,N)
xx, yy = np.meshgrid(x,y)
z = np.zeros((K**2, xx.shape[0], xx.shape[1]))
mux = np.linspace(-1,1,K)
muy = np.linspace(-1,1,K)
mmx, mmy = np.meshgrid(mux, muy)
mm = np.column_stack((mmx.ravel(), mmy.ravel()))
for i in range(K**2):
    z[i,:,:] = amp/(2*np.pi*sd**2) * np.exp((-1/2)*(((xx-mm[i,0])/sd)**2 +
        ((yy-mm[i,1])/sd)**2)**2)

data = np.reshape(z, (K**2, N**2))    # Matrix with each row a data point.
data = np.vstack(data, np.zeros(N**2))
plt.imshow(z[3,:,:], cmap='gray')
plt.show()

# Compute the distance matrix and persistence.
D = sp.spatial.distance.cdist(data, data, metric='euclidean')
PH = ripser(D, distance_matrix=True, maxdim=2)
plot_diagrams(PH['dgms'])
plt.show()
# If the parameters are chosen well the PH should have a nice H^2 class.
