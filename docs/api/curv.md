# Curvature Module

The curvature module provides implementations for various curvature approximations and matrix-vector products used in Laplace approximation.

## Overview


### Curvatures
Currently supported curvatures are the:

- **GGN (Generalized Gauss-Newton)**: Efficient matrix-vector products for neural networks
- **HVP (Hessian)**: Efficient matrix-vector products for neural networks weight Hessian.

Both aim to support a data batch or a data loader.

### Curvature approximations
Currently the following curvature approximations and posterior derviations are supported:

- **Full** Full representation of the matrix
- **Diagonal** Diagonal approximation of any curvature structure.
- **Low Rank** Low-rank approximation of any curvature structure.

Each method leads to a corresponding weight space covariance matrix-vector product. Additional curvature can be easily registered to have the same pipeline available.

## Key functions

All functionalities are summarized in the following function, that takes a curvature-vector-product and returns a function that maps `prior_arguments` to the posterior.

::: laplax.curv.cov.create_posterior_function

## Bonus

Additional curvature approximations/pipelines can be registered with the following function:

::: laplax.curv.cov.register_curvature_method
