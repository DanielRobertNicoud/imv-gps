# Intrinsic manifold vector GPs
Code for the paper "Intrinsic Gaussian Vector Fields on Manifolds" by Robert-Nicoud, Krause, and Borovitskiy.

**Note of the author:**
* The code extends `scikit-learn`'s GP implementation. It is a bit of a hack and it is a bit messy, but it works.s A cleaner version will soon be available in the [https://github.com/GPflow/GeometricKernels/tree/main](GeometricKernels) package.
* Beware: this repository contains a 243MB data file to allow reproducing the experiments.

## Scripts present
1. `aux_functions.py`: Auxiliary functions used to treat ponts and vectors on the sphere.
2. `sphere_vector_kernel.py`: Implementation of vector kernels on the sphere extending `sklearn.gaussian_process.kernels.Kernel`.
3. `sphere_vector_gp.py`: Implementation of manifold vector GPs extending `sklearn.gaussian_process.GaussianProcessRegressor`.
4. `blender_file_generation.py`: Utility for saving outputs ready to be treated by blender.
5. `001_gp_prior_samples`.ipynb: Generate samples from GP priors with various kernels.
6. `002_blender_eigenvf.ipynb`: Some heat equation eigen-vector fields on the sphere.
7. `003_era5_experiments.ipynb`: Run GP experiments on the ERA5 data.
8. `.ipynb`:
9. `.ipynb`:
10. `006_var_div.ipynb`: Computation of variance of the divergence of various GPs.
