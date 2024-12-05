"""Handles the actual low-rank approximation from a matrix-vector product."""

import jax
import jax.numpy as jnp
from jax.experimental.sparse.linalg import lobpcg_standard

from laplax.types import Callable, DType, KeyType

# ---------------------------------------------------------------
# Low Rank Approximations
# ---------------------------------------------------------------


def get_low_rank_large_eigenvalues(
    mv: Callable,
    key: KeyType,
    size: int,
    maxiter: int,
    dtype: DType = jnp.float64,
) -> dict:
    """Computes a low-rank approximation focusing on the largest eigenvalues.

    Args:
        mv (Callable): Matrix-vector product function.
        key (KeyType): PRNG key for random initialization.
        size (int): Size of the input/output space.
        maxiter (int): Number of iterations for the approximation.
        dtype (DType): Data type for computations.

    Returns:
        dict: A dictionary containing:
            - "U": Eigenvectors as a matrix.
            - "S": Corresponding eigenvalues.
    """
    # Ensure maxiter is appropriate for the size
    if size < maxiter * 5:
        maxiter = max(1, size // 5 - 1)
        print(f"Warning: Reduced maxiter to {maxiter} due to insufficient size.")  # noqa: T201

    # Initialize random basis
    b = jax.random.normal(key, (size, maxiter), dtype=dtype)

    # Perform LOBPCG for eigenvalues and eigenvectors
    eigenvals, eigenvecs, _ = lobpcg_standard(mv, b, m=maxiter)

    return {
        "U": jnp.asarray(eigenvecs, dtype=dtype),
        "S": jnp.asarray(eigenvals, dtype=dtype),
    }


def get_low_rank(  # noqa: D417, PLR0913, PLR0917
    mv: Callable,
    key: KeyType,
    size: int,
    maxiter: int = 20,
    dtype: DType = jnp.float64,
    enable_local_dtype_switch: bool = True,  # noqa: FBT001, FBT002
    **kwargs,  # noqa: ARG001
) -> dict:
    """Handles low-rank approximations with optional precision adjustments.

    Args:
        mv (Callable): Matrix-vector product function.
        key (KeyType): PRNG key for random initialization.
        size (int): Size of the input/output space.
        maxiter (int): Number of iterations for the approximation (default: 20).
        dtype (DType): Desired data type for computations (default: float64).
        enable_local_dtype_switch (bool): If True, temporarily enables higher precision.

    Returns:
        dict: A dictionary containing:
            - "U": Eigenvectors as a matrix.
            - "S": Corresponding eigenvalues.
    """
    # Configure local dtype if necessary
    original_dtype_config = jax.config.read("jax_enable_x64")
    local_dtype = dtype

    if enable_local_dtype_switch and dtype != jnp.float64:
        jax.config.update("jax_enable_x64", True)  # noqa: FBT003
        local_dtype = jnp.float64

    # Compute low-rank approximation
    low_rank_result = get_low_rank_large_eigenvalues(
        mv=mv,
        key=key,
        size=size,
        maxiter=maxiter,
        dtype=local_dtype,
    )

    # Convert back to desired dtype if different from local dtype
    if local_dtype != dtype:
        low_rank_result = jax.tree_map(lambda x: x.astype(dtype), low_rank_result)

    # Restore original dtype configuration if it was changed
    if enable_local_dtype_switch and original_dtype_config != jax.config.read(
        "jax_enable_x64"
    ):
        jax.config.update("jax_enable_x64", original_dtype_config)

    return low_rank_result
