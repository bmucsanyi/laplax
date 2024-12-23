"""Higher-level API for low_rank approximations.

This module provides utilities for computing low-rank approximations using
the Locally Optimal Block Preconditioned Conjugate Gradient (LOBPCG) algorithm
with mixed-precision support.

The primary function, `get_low_rank_approximation`, computes a low-rank
approximation of a matrix using a matrix-vector product function. It supports
customizable data types, tolerance levels, and the ability to disable JIT
compilation when required.
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
    """Dataclass containing the components of the low-rank curvature approximation.

    Attributes:
        U: The eigenvectors matrix.
        S: The eigenvalues array.
        scalar: The scalar factor.
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
    """Computes a low-rank approximation using the LOBPCG algorithm.

    This function computes the leading eigenvalues and eigenvectors of a matrix
    represented by a matrix-vector product function `mv`. It supports mixed-precision
    arithmetic and optional JIT compilation.

    Args:
        mv (Callable[[jax.Array], jax.Array]):
            Matrix-vector product function representing the matrix A(x).
        key (KeyType):
            PRNG key for random initialization of search directions.
        size (int):
            Dimension of the input/output space.
        maxiter (int, optional):
            Maximum number of LOBPCG iterations. Default is 20.
        mv_dtype (DType, optional):
            Data type for matrix-vector product calls. Default is float32.
        calc_dtype (DType, optional):
            Data type for internal calculations during LOBPCG. Default is float64.
        return_dtype (DType, optional):
            Desired output data type for results. Default is float32.
        tol (float | None, optional):
            Tolerance for convergence. If None, uses default machine epsilon.
        mv_jittable (bool, optional):
            If True, enables JIT compilation for LOBPCG. Default is True.
        **kwargs: Not needed.

    Returns:
        LowRankTerms: A dataclass containing:
            - "U" (jax.Array): Eigenvectors as a matrix of shape `(size, maxiter)`.
            - "S" (jax.Array): Corresponding eigenvalues as a vector of length
                `maxiter`.
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
