# Graph-Based Classification via Diffuse Interface Models

An implementation of semi-supervised classification on graphs using a
Ginzburg-Landau diffuse-interface energy model, minimised via convex
splitting in a low-dimensional graph Laplacian eigenbasis.

This is a from-scratch implementation of the method described in
Bertozzi & Flenner, *"Diffuse Interface Models on Graphs for
Classification of High Dimensional Data"* (SIAM Review, 2016), written
alongside my Master's thesis on the same topic at the University of
Koblenz. It's a clean demonstration of the algorithm on a public toy
dataset, not the thesis's internal codebase.

## How it works

1. **Graph construction** (`graph_utils.py`): builds a symmetric k-nearest-neighbour
   graph with Gaussian similarity weights, and computes the (normalised)
   graph Laplacian.
2. **Spectral basis**: the classification problem is projected onto the
   leading eigenvectors of the graph Laplacian, drastically reducing the
   dimensionality of the optimisation.
3. **Diffuse interface energy** (`diffuse_interface_classifier.py`): relaxes
   the discrete graph-cut classification problem into a continuous
   Ginzburg-Landau energy functional, with a fidelity term that anchors
   the small set of known-labelled points.
4. **Convex splitting**: the energy is minimised iteratively via a
   convex-splitting numerical scheme, which is unconditionally stable for
   this type of non-convex energy.
5. **Classification**: the sign of the final continuous field gives the
   predicted class for every point in the dataset, including the
   unlabelled ones.

## Demo: Two Moons dataset

`run_two_moons_demo.py` classifies the classic "Two Moons" dataset using
only 6 labelled points out of 300 (3 per class), and correctly classifies
the rest with ~84% accuracy purely from graph structure.

## Tools used

Python, NumPy, SciPy (k-NN distances, eigendecomposition), scikit-learn
(dataset generation), Matplotlib (visualisation)

## How to run

```bash
pip install numpy scipy scikit-learn matplotlib
python run_two_moons_demo.py
```

## Background

This method is the core of my Master's thesis, *"Classification of
High-Dimensional Data on Graphs using Diffuse Interface Models and Graph
Laplacians"*, at the University of Koblenz, where I extend this approach
to higher-dimensional real-world datasets and analyse parameter
sensitivity and classification accuracy in depth.

## Reference

Bertozzi, A. L., & Flenner, A. (2016). Diffuse Interface Models on Graphs
for Classification of High Dimensional Data. *SIAM Review*, 58(2), 293-328.
