"""Higher-level API for Low-Rank Approximations.

This module provides utilities for computing low-rank approximations using
the Locally Optimal Block Preconditioned Conjugate Gradient (LOBPCG) algorithm.
It supports mixed-precision arithmetic, customizable data types, and optional
JIT compilation for optimized performance.

The primary function, `get_low_rank_approximation`, computes the leading eigenvalues
and eigenvectors of a matrix represented by a matrix-vector product function.
This allows for scalable computation without explicitly constructing the full
matrix, making it efficient for large-scale problems.

Key Features:
- Mixed-precision support for reduced memory usage and improved performance.
- Flexible tolerance and iteration settings for adaptive convergence.
- JIT compilation for efficient matrix-vector product computations.
"""

import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from laplax.curv.lanczos import lobpcg_standard
from laplax.types import Array, Callable, DType, Float, KeyType, Num
from laplax.util.flatten import wrap_function

# -----------------------------------------------------------------------------
# Low-rank terms
# -----------------------------------------------------------------------------


@dataclass
class LowRankTerms:
    """Components of the low-rank curvature approximation.

    This dataclass encapsulates the results of the low-rank approximation, including
    the eigenvectors, eigenvalues, and a scalar factor which can be used for the prior.

    Attributes:
        U: Matrix of eigenvectors, where each column corresponds to an eigenvector.
        S: Array of eigenvalues associated with the eigenvectors.
        scalar: Scalar factor added to the matrix during the approximation.
    """

    U: Num[Array, "P R"]
    S: Num[Array, " R"]
    scalar: Float[Array, ""]


jax.tree_util.register_pytree_node(
    LowRankTerms,
    lambda node: ((node.U, node.S, node.scalar), None),
    lambda _, children: LowRankTerms(U=children[0], S=children[1], scalar=children[2]),
)


# -----------------------------------------------------------------------------
# Low-rank approximation
# -----------------------------------------------------------------------------


def get_low_rank_approximation(
    mv: Callable[[Array], Array],
    key: KeyType,
    size: int,
    maxiter: int = 20,
    mv_dtype: DType = jnp.float32,
    calc_dtype: DType = jnp.float64,
    return_dtype: DType = jnp.float32,
    tol: float | None = None,
    *,
    mv_jittable: bool = True,
    **kwargs,
) -> LowRankTerms:
    r"""Compute a low-rank approximation using the LOBPCG algorithm.

    This function computes the leading eigenvalues and eigenvectors of a matrix
    represented by a matrix-vector product function `mv`, without explicitly forming
    the matrix. It uses the Locally Optimal Block Preconditioned Conjugate Gradient
    (LOBPCG) algorithm to achieve efficient low-rank approximation, with support
    for mixed-precision arithmetic and optional JIT compilation.

    Mathematically, the low-rank approximation seeks to find the leading eigenpairs
    $(\lambda_i, u_i)$ such that:
    $A u_i = \lambda_i u_i \quad \text{for } i = 1, \ldots, k$, where $A$ is the matrix
    represented by the matrix-vector product `mv`, and $k$ is the number of eigenpairs.

    Args:
        mv: A callable that computes the matrix-vector product, representing the matrix
            $A(x)$.
        key: PRNG key for random initialization of the search directions.
        size: Dimension of the input/output space of the matrix.
        maxiter: Maximum number of LOBPCG iterations. Defaults to 20.
        mv_dtype: Data type for the matrix-vector product function.
        calc_dtype: Data type for internal calculations during LOBPCG.
        return_dtype: Data type for the final results.
        tol: Convergence tolerance for the algorithm. If `None`, the machine epsilon
            for `calc_dtype` is used.
        mv_jittable: If `True`, enables JIT compilation for the matrix-vector product.
        **kwargs: Additional arguments (ignored).

    Returns:
        LowRankTerms: A dataclass containing:
            - `U`: Eigenvectors as a matrix of shape `(size, rank)`.
            - `S`: Eigenvalues as an array of length `rank`.
            - `scalar`: Scalar factor, initialized to 0.0.

    Raises:
        ValueError: If `size` is insufficient to perform the requested number of
            iterations.

    Notes:
        - If the size of the matrix is small relative to `maxiter`, the number of
          iterations is reduced to avoid over-computation.
        - Mixed precision can significantly reduce memory usage, especially for large
          matrices.

    Example:
        ```python
        def mv_function(x):
            return A @ x  # Replace A with your matrix or matrix representation

        low_rank_terms = get_low_rank_approximation(
            mv=mv_function,
            key=jax.random.PRNGKey(42),
            size=1000,
            maxiter=10,
            tol=1e-6,
        )
        ```
    """
    del kwargs

    # Adjust maxiter if it's too large compared to problem size
    if size < maxiter * 5:
        maxiter = max(1, size // 5 - 1)
        msg = f"Reduced maxiter to {maxiter} due to insufficient size."
        warnings.warn(msg, stacklevel=1)

    is_compute_in_float64 = jax.config.read("jax_enable_x64")
    if jnp.float64 in {mv_dtype, calc_dtype, return_dtype}:
        jax.config.update("jax_enable_x64", True)

    # Wrap to_dtype around mv if necessary.
    if mv_dtype != calc_dtype:
        mv = wrap_function(
            mv,
            input_fn=lambda x: jnp.asarray(x, dtype=mv_dtype),
            output_fn=lambda x: jnp.asarray(x, dtype=calc_dtype),
        )

    # Initialize random search directions
    X = jax.random.normal(key, (size, maxiter), dtype=calc_dtype)

    # Perform LOBPCG for eigenvalues and eigenvectors using the new wrapper
    eigenvals, eigenvecs, _ = lobpcg_standard(
        A=mv,
        X=X,
        m=maxiter,
        tol=tol,
        calc_dtype=calc_dtype,
        a_dtype=mv_dtype,
        A_jittable=mv_jittable,
    )

    # Prepare and convert the results
    low_rank_result = LowRankTerms(
        U=jnp.asarray(eigenvecs, dtype=return_dtype),
        S=jnp.asarray(eigenvals, dtype=return_dtype),
        scalar=jnp.asarray(0.0, dtype=return_dtype),
    )

    # Restore the original configuration dtype
    if is_compute_in_float64 != jax.config.read("jax_enable_x64"):
        jax.config.update("jax_enable_x64", is_compute_in_float64)

    return low_rank_result
