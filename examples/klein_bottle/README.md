# Summary of Experiments

Datasets:
* Line images, a model of $\RP^2$.
* Colored line images, a model of $\RP^3$.
* Point light-source images, a model of $\CP^1$.
* The flat Klein bottle. (Or the image version?)
* A curve in the space of line images.

Files:
* `klein_bottle_experiment.ipynb`



We compare the following methods on each dataset:
* Classical PCA/MDS,
* Projective PCA (iterative),
* Projective PCA (batch),
* Projective MDS (chordal),
* Projective MDS (geodesic),
* Projective MDS (geodesic with PCA initial condition).

The comparison metrics used are the following:
* Metric distortion in goal dimension, as $\frac{\| D - D_{\mathrm{goal}} \|}{\| D_{\mathrm{goal}}}$.
* Accuracy of topology in goal dimension (qualitatively via persistence diagrams, and qunatified by bottleneck/Wasserstein distance computations when appropriate).
* Minimum dimension in which correct topology could be achieved.
* Execution time of algorithm.

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

## Conclusions:

### PCA:
Reasonable embedding requires Euclidean dimension eight or higher. Six looks okay, but seven is weird. Pretty stable with eight and above. Quantitative measures of difference not reasonable.

### Projective PCA:
Embeddings look good. They recover the Möbius band, as hoped for on RP2. Both iterative and batch get about the same result.
