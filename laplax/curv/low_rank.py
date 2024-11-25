from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.experimental.sparse.linalg import lobpcg_standard

from laplax.types import DType, KeyType

# ---------------------------------------------------------------
# Low rank approximations
# ---------------------------------------------------------------


def get_low_rank_large_eigenvalues(
    mv: Callable,
    key: KeyType,
    maxiter: int,
    size: int,
    dtype: DType = jnp.float64,
) -> dict:
    # Assert
    if size < maxiter * 5:
        print(
            f"Warning: maxiter {maxiter} too large for size, "
            f"reducing to {size // 5 - 1}."
        )
        maxiter = size // 5 - 1

    # Initialize lanczos
    b = jax.random.normal(key, (size, maxiter), dtype=dtype)

    # Run lanczos
    eigenvals, eigenvecs, _ = lobpcg_standard(mv, b, m=maxiter)

    return {
        "U": jnp.asarray(eigenvecs, dtype=dtype),
        "S": jnp.asarray(eigenvals, dtype=dtype),
    }


def get_low_rank(
    mv: Callable,
    key: KeyType,
    size: int,
    maxiter: int = 20,
    local_dtype_switch: bool = True,
    dtype: DType = jnp.float64,
    **kwargs,
) -> dict:
    """Handling low rank calculations."""
    # Set local dtype if necessary
    if local_dtype_switch and dtype != jnp.float64:
        jax.config.update("jax_enable_x64", True)
        local_dtype = jnp.float64
    else:
        local_dtype = dtype

    # Get low rank
    low_rank_tuple = get_low_rank_large_eigenvalues(
        mv, maxiter=maxiter, size=size, key=key, dtype=local_dtype
    )

    # Adjust dtype and set global dtype variable.
    if dtype != local_dtype:
        low_rank_tuple = jax.tree.map(dtype, low_rank_tuple)

        if local_dtype_switch:
            jax.config.update("jax_enable_x64", False)

    return low_rank_tuple
