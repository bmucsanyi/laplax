# Curvature Approximations

### Full approximation

This essentially denses the curvature matrix-vector product to a full array and uses it as the covariance precision for creating the weight space posterior covariance.

::: laplax.curv.cov.create_full_curvature
    options:
      show_root_heading: true
      show_source: true
      members_order: source


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
