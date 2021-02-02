# Uniform S^2

This folder contains examples for testing _complex projective MDS_. This is a
toy example, intended to verify that one configuration of points (the "knotted
sphere") can be correctly moved to a specified configuration (the uniform
sphere). It does not perform any actual dimensionality reduction.

Scripts:
* `uniform_sphere.py`: Generates data uniformly at random on an n-sphere.
* `knotted_two_sphere.py`: Forms the suspension of a trefoil not, which is a
  "knotted" 2-sphere in $R^4$.
* `knotted_two_sphere_R8.py`: The same construction, but with extra coordinates
  placing it in $R^8 = C^4$. This is not "knotted" in any real sense.
* `cpn_script.py`: Loads the data file `cp_test*.npz` and runs complex
  projective MDS on it. This mostly exists as a way to run this experiment on
  the HPCC.

Data Files:
* `cp_test144.npz`: Contains a distance matrix corresponding to 144 points
  sampled uniformly at random from S^2, and a configuration of points on `CP^1`
  constructed as the suspension of the trefoil knot in `R^3`.
* `cp_test576.npz`: Same as above, but corresponding to 576 points. This one
  took too long to run in initial testing.
* `output_N144_lam1.npz`: Contains a point cloud `Xopt` and the real and
  imaginary parts of the roots of unity giving the inner product, `Areal` and
  `Aimag`. This run was performed with a penalty term of lambda = 1.
* `output_N144_lam10.npz`: Same as above with penalty lambda = 10.
* `output_N144_lam100.npz`: Penalty lambda = 100. This was the best example.
* `output_N144_lam1000.npz`: Penalty lambda = 1000.

Output Files:
* `PH_goal.png`, `PH_initial.png`, `PH_lam1.png`, `PH_lam100.png`: Persistence
  diagrams for the original distance matrix, the starting configuration of
  points, and the results of the lambda=1 and lambda=100 runs.
* `points_initial.png`, `points_lam1.png`, `points_lam100.png`: Scatter plots
  of the initial configuration and the output from the lambda=1 and lambda=100
  runs.
