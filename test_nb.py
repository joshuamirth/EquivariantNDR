# Embed a circle in RP^4 with five "kinks".
import testing_utils
import ppca
X,n = testing_utils.circleRPn()

# Do a dimensionality reduction from X to RP^2 using PPCA.
V = ppca.ppca(X, 2)
P = V['X']
testing_utils.plot_RP2(P)

# Do the dimensionality reduction from X to RP^2 using PMDS.
M = testing_utils.pmds(X,2)
testing_utils.plot_RP2(M)

# Now do the same with RP^3 and four "kinks".
X,n = testing_utils.circleRPn(dimn=3,num_segments=3)

# Do a dimensionality reduction from X to RP^2 using PPCA.
V = ppca.ppca(X, 2)
P = V['X']
testing_utils.plot_RP2(P)

# Do the dimensionality reduction from X to RP^2 using PMDS.
M = testing_utils.pmds(X,2)
testing_utils.plot_RP2(M)
