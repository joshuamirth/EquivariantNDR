# Point Image $CP^1$

This example generates a collection of images of a single, blurry, bright point
against a dark background. This is a representation of $CP^1$ since it can be
thought of as the plane plus a point at infinity (when the bright point has
moved completely out of frame). 

The pipeline for this example is as follows:
* Generate a collection of images.
* Compute persistent homology of a landmark subset of these images.
* Use a Bockstein lift to get integer coordinates in PH_2.
* Map into $CP^N$ where $N$ is the number of landmarks.
* Perform PPCA to get a low-dimensional embedding.
* Apply MDS to optimize the embedding. The final optimum should hopefully be a
  uniform sampling from $CP^1$.
  
Currently this is a work in progress.
