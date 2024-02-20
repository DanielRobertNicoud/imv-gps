# Intrinsic manifold vector GPs
Code for the paper "Intrinsic Gaussian Vector Fields on Manifolds" by Robert-Nicoud, Krause, and Borovitskiy.

**Note of the author:**
* The code extends `scikit-learn`'s GP implementation. It is a bit of a hack and it is a bit messy, but it works.s A cleaner version will soon be available in the [https://github.com/GPflow/GeometricKernels/tree/main](GeometricKernels) package.
* The ERA5 dataset is too large to be uploaded. It can be freely downloaded at
  https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels-monthly-means?tab=form
  by selecting:
  * Product type -> Reanalysis
  * Variable -> U-component of wind + V-component of wind
  * Pressure level -> 500 hPa
  * Year -> 2010
  * Month -> all available
  * Time -> 00:00
  * Format -> NetCDF
  It should then be uploaded in the `era5` folder.
* The numbered scripts should be ran in order, as some generate data that is consuimed by the ones following.

## Scripts present
1. `aux_functions.py`: Auxiliary functions used to treat ponts and vectors on the sphere.
2. `sphere_vector_kernel.py`: Implementation of vector kernels on the sphere extending `sklearn.gaussian_process.kernels.Kernel`.
3. `sphere_vector_gp.py`: Implementation of manifold vector GPs extending `sklearn.gaussian_process.GaussianProcessRegressor`.
4. `blender_file_generation.py`: Utility for saving outputs ready to be treated by blender.
5. `001_gp_prior_samples`.ipynb: Generate samples from GP priors with various kernels.
6. `002_blender_eigenvf.ipynb`: Some heat equation eigen-vector fields on the sphere.
7. `003_era5_experiments.ipynb`: Run GP experiments on the ERA5 data.
8. `004_flat_plots.ipynb`: Shows some of the results in paper-grade quality on projected maps.
9. `.ipynb`:
10. `006_var_div.ipynb`: Computation of variance of the divergence of various GPs.
