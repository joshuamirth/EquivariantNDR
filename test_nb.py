# Embed a circle in RP^4 with five "kinks".
import projective_mds
import ppca
X,n = projective_mds.circleRPn()

# Do a dimensionality reduction from X to RP^2 using PPCA.
V = ppca.ppca(X, 2)
P = V['X']
projective_mds.plot_RP2(P)

# Do the dimensionality reduction from X to RP^2 using PMDS.
M = projective_mds.pmds(X,2)
projective_mds.plot_RP2(M)

# Now do the same with RP^3 and four "kinks".
X,n = projective_mds.circleRPn(dimn=3,num_segments=3)

# Do a dimensionality reduction from X to RP^2 using PPCA.
V = ppca.ppca(X, 2)
P = V['X']
projective_mds.plot_RP2(P)

# Do the dimensionality reduction from X to RP^2 using PMDS.
M = projective_mds.pmds(X,2)
projective_mds.plot_RP2(M)
