# Embed a circle in RP^5 with two "kinks".
import testing_utils
import ppca
X = testing_utils.circleRPn()

# Do a dimensionality reduction from X to RP^2 using PPCA.
V = ppca.ppca(X, 2)
P = V['X']
testing_utils.plot_RP2(P)

# Do the dimensionality reduction from X to RP^2 using PMDS.
M = testing_utils.pmds(X,2)
testing_utils.plot_RP2(M)
