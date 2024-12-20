# Curvature Approximations

### Exact Representation

Provides the dense representation of the curvature matrix-vector product.
It is used for creating the weight space posterior covariance without further approximations.

::: laplax.curv.cov.create_full_curvature
    options:
      show_root_heading: true
      show_source: true
      members_order: source


### Low Rank Approximations

Provides an efficient low-rank approximation of curvature matrices.

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
