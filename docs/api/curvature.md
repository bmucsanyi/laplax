# Curvature Module

The curvature module provides implementations for various curvature approximations and matrix-vector products used in Laplace approximation.

## Overview


### Curvature methods
Currently supported curvatures are the:

- **GGN (Generalized Gauss-Newton)**: Efficient matrix-vector products for neural networks
- **HVP (Hessian)**: Efficient matrix-vector products for neural networks weight Hessian. 

Both aim to support a data batch or a data loader.

### Curvature approximations
Currently the following curvature approximations and posterior derviations are supported:

- **Full** Full representation of the matrix
- **Diagonal** Diagonal approxmation of any curvature structure.
- **Low Rank** Low-rank approximation of any curvature structure.

Each method leads to a corresponding weight space covariance matrix-vector product. Additional curvature can be easily registered to have the same pipeline available.


## Curvatures

### Generalized Gauss-Newton (GGN)

The GGN module provides efficient implementations of Generalized Gauss-Newton matrix-vector products.

::: laplax.curv.ggn.create_ggn_mv
    options:
      show_root_heading: true
      show_source: true
      members_order: source

::: laplax.curv.ggn.create_ggn_mv_without_data
    options:
      show_root_heading: true
      show_source: true
      members_order: source

::: laplax.curv.ggn.create_loss_hessian_mv
    options:
      show_root_heading: true
      show_source: true
      members_order: source



## Curvature Approximations

### Full approxmation



### Low Rank Approximations

Tools for computing efficient low-rank approximations of curvature matrices.

::: laplax.curv.low_rank.get_low_rank_approximation
    options:
      show_root_heading: true
      show_source: true
      members_order: source
 

A mixed-precision implementation of the LOBPCG algorithm for eigenvalue computation.

::: laplax.curv.lanczos.lobpcg_standard
    options:
      show_root_heading: true
      show_source: true
      members_order: source

## Covariance Functions

Functions for posterior covariance estimation with different approximation methods.

### Factory Functions

::: laplax.curv.cov.create_posterior_function
    options:
      show_root_heading: true
      show_source: true
      members_order: source

### Curvature Methods

::: laplax.curv.cov.create_full_curvature
    options:
      show_root_heading: true
      show_source: true

::: laplax.curv.cov.create_diagonal_curvature
    options:
      show_root_heading: true
      show_source: true

::: laplax.curv.cov.create_low_rank_curvature
    options:
      show_root_heading: true
      show_source: true

### Registration

::: laplax.curv.cov.register_curvature_method
    options:
      show_root_heading: true
      show_source: true
      members_order: source

