# Klein Bottle

This is an example for _real projective MDS_. The basic data set is the flat
model of the Klein bottle.

The pipeline for this example is as follows:
* Generate data on the flat Klein bottle.
* Compute persistence of a landmark subset thereof.
* Map into `RP^N` where `N` is the number of landmarks.
* Reduce dimension to `RP^2` using PPCA.
* Optimize with MDS.

In principle this example could also be used to illustrate coordinates in `L_4`
by applying a lift to the cohomology classes into `Z_4`. There is theory to
suggest there should be a nice embedding there.

Scripts
Data Files
Output Files
