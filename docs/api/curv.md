# Curvature Module

The curvature module provides implementations for various curvature approximations and matrix-vector products used in Laplace approximations.

## Overview


### Curvatures
Currently supported curvatures:

- **GGNVP (Generalized Gauss-Newton):** Efficient matrix-vector products for neural network Generalized Gauss-Newton (GGN) matrices.
- **HVP (Hessian):**: Matrix-vector products for neural network loss Hessians.

Both of these curvatures support a mini-batch of data or a data loader.

### Curvature approximations
Currently, the following approximations and corresponding weight posteriors are supported for any of the curvatures:

- **Full:** Exact representation of the curvature.
- **Diagonal:** Diagonal approximation of the curvature.
- **Low Rank** Low-rank approximation of the curvature.

Each method leads to a corresponding weight-space covariance matrix-vector product. Custom curvatures can be easily registered and used in our pipelines.
