# Curvature Approximations

### Exact Representation

Provides the dense representation of the curvature matrix-vector product.
It is used for creating the weight space posterior covariance without further approximations.

::: laplax.curv.cov.create_full_curvature

::: laplax.curv.cov.full_with_prior

::: laplax.curv.cov.full_prec_to_state

::: laplax.curv.cov.full_state_to_scale

::: laplax.curv.cov.full_state_to_cov

### Diagonal

The diagonal approximation represents the curvature matrix using only its diagonal elements, providing a computationally efficient but simplified representation of the curvature information.

::: laplax.curv.cov.create_diagonal_curvature

::: laplax.curv.cov.diag_with_prior

::: laplax.curv.cov.diag_prec_to_state

::: laplax.curv.cov.diag_state_to_scale

::: laplax.curv.cov.diag_state_to_cov

### Low Rank Approximations

Provides an efficient low-rank approximation of curvature matrices.

::: laplax.curv.low_rank.get_low_rank_approximation

::: laplax.curv.cov.create_low_rank_curvature

::: laplax.curv.cov.create_low_rank_mv

::: laplax.curv.cov.low_rank_with_prior

::: laplax.curv.cov.low_rank_prec_to_state

::: laplax.curv.cov.low_rank_state_to_scale

::: laplax.curv.cov.low_rank_state_to_cov


A mixed-precision implementation of the LOBPCG algorithm for eigenvalue computation.

::: laplax.curv.lanczos.lobpcg_standard
    options:
      show_root_heading: true
      show_source: true
      members_order: source
