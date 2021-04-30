# Summary of Experiments

Datasets:
* Line images, a model of $\RP^2$.
* Colored line images, a model of $\RP^3$.
* Point light-source images, a model of $\CP^1$.
* The flat Klein bottle. (Or the image version?)
* A curve in the space of line images.


We compare the following methods on each dataset:
* Classical PCA/MDS,
* Projective PCA (iterative),
* Projective PCA (batch),
* Projective MDS, which has the following variations:
    - True geodesic metric, with random and PCA initial conditions.
    - Squared geodesic metric, with random and PCA initial conditions.
    - True chordal metric, with random and PCA initial conditions.
    - Squared chordal metric, with random and PCA initial conditions.

The comparison metrics used are the following:
* Minimum dimension in which correct topology could be achieved. For some methods/datasets there will be no such dimension. Classical PCA should be the baseline here.
* Metric distortion in goal dimension, as `||D - D_G|| / ||D_G||`.
* Accuracy of topology in goal dimension (qualitatively via persistence diagrams, and quantified by bottleneck/Wasserstein distance computations when appropriate). This may not be necessary since I expect that getting the correct topology in the goal dimension is an either/or outcome and metric distortion is more useful than bottleneck distances.
* Execution time of algorithm. This is primarily for the true/squared MDS methods. Should also record required number of iterations. If true is significantly better on other metrics we may just ignore squared methods rendering this useless.

# Klein Bottle

Files:
* `klein_bottle_experiment.ipynb`

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

## MDS

Story is that we can perform gradient descent directly, but may achieve equal quality results with some small approximations that hopefully improve performance significantly.

### MDS with Squared Geodesic Distance

PCA (Batch) Initial condition

Dimension refers to the ambient dimension. Projective space is one less.

Dimension | Iterations | Time (seconds) | Correct Topology | Metric Distortion | Bottleneck to Goal | Wasserstein to Goal |
--------- | ---------- | -------------- | ---------------- | ----------------- | ------------------ | ------------------- |
2         |  25        | 5.18           | No               |                   |                    |                     | 
3         |  56        | 10.64          | No               |                   |                    |                     | 
4         |  78        | 15.38          | No               |                   |                    |                     | 
5         |  46        | 10.70          | Yes              |                   |                    |                     | 
6         |  56        | 13.65          | Yes              |                   |                    |                     | 
7         |  47        | 10.49          | Yes              |                   |                    |                     | 
8         |  51        | 10.71          | Yes              |                   |                    |                     | 
9         |  52        | 11.35          | Yes              |                   |                    |                     | 
10        |  48        | 11.45          | Yes              |                   |                    |                     | 
11        |  49        | 10.55          | Yes              |                   |                    |                     | 

Random Initial condition

Dimension | Iterations | Time (seconds) | Correct Topology | Metric Distortion | Bottleneck to Goal | Wasserstein to Goal |
--------- | ---------- | -------------- | ---------------- | ----------------- | ------------------ | ------------------- |
2         |  89        | 19.9           | No               |                   |                    |                     |
3         |  96        | 21.52          | No               |                   |                    |                     |
4         |  99        | 21.71          | No               |                   |                    |                     |
5         |  89        | 18.96          | No               |                   |                    |                     |
6         |  98        | 19.26          | Maybe Yes        |                   |                    |                     |
7         |  147       | 27.74          | No               |                   |                    |                     |
8         |  98        | 20.29          | No               |                   |                    |                     |
9         |  126       | 25.75          | No               |                   |                    |                     |
10        |  195       | 38.12          | No               |                   |                    |                     |
11        |  142       | 26.89          | Maybe Yes        |                   |                    |                     |

### MDS with Weighted Norm Approximation

PCA IC

Dimension | Iterations | Time (seconds) | Correct Topology | Metric Distortion | Bottleneck to Goal | Wasserstein to Goal |
--------- | ---------- | -------------- | ---------------- | ----------------- | ------------------ | ------------------- |
2         |  38        | 9.62           |                  |                   |                    |                     | 
3         |  79        | 17.78          |                  |                   |                    |                     | 
4         |  194       | 49.29          |                  |                   |                    |                     | 
5         |  58        | 15.11          |                  |                   |                    |                     | 
6         |  58        | 14.43          |                  |                   |                    |                     | 
7         |  56        | 14.29          |                  |                   |                    |                     | 
8         |  60        | 14.62          |                  |                   |                    |                     | 
9         |  51        | 12.23          |                  |                   |                    |                     | 
10        |  58        | 15.07          |                  |                   |                    |                     | 
11        |  53        | 12.99          |                  |                   |                    |                     | 

Random Initial condition

Dimension | Iterations | Time (seconds) | Correct Topology | Metric Distortion | Bottleneck to Goal | Wasserstein to Goal |
--------- | ---------- | -------------- | ---------------- | ----------------- | ------------------ | ------------------- |
2         |  67        | 17.28          |                  |                   |                    |                     |
3         |  744       | 185.24         |                  |                   |                    |                     |
4         |  822       | 208.07         |                  |                   |                    |                     |
5         |  681       | 170.64         |                  |                   |                    |                     |
6         |  549       | 159.01         |                  |                   |                    |                     |
7         |  697       | 199.22         |                  |                   |                    |                     |
8         |  802       | 224.40         |                  |                   |                    |                     |
9         |  1096      | 290.79         |                  |                   |                    |                     |
10        |  1403      | 387.39         |                  |                   |                    |                     |
11        |  1275      | 365.39         |                  |                   |                    |                     |



### MDS with no approximations