from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.experimental.sparse.linalg import lobpcg_standard

from laplax.curv.lanczos import lanczos_isqrt_full_reortho, lanczos_random_init
from laplax.types import DType, KeyType

# ---------------------------------------------------------------
# Low rank approximations
# ---------------------------------------------------------------


def get_low_rank_small_eigenvalues(
    mv: Callable,
    maxiter: int,
    size: int,
    key: KeyType,
    dtype: DType = jnp.float64,
) -> dict:
    # Initialize the Lanczos algorithm
    b = lanczos_random_init(key, size)

    # Run the Lanczos algorithm
    D = lanczos_isqrt_full_reortho(mv, b, maxiter=maxiter)

    # Calculate svd of low rank
    svd_result = jnp.linalg.svd(D, full_matrices=False)

    return {
        "U": jnp.asarray(svd_result.U, dtype=dtype),
        "S": jnp.asarray(svd_result.S**-2, dtype=dtype),
    }


def get_low_rank_large_eigenvalues(
    mv: Callable,
    maxiter: int,
    size: int,
    key: KeyType,
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
        "S": jnp.asarray(eigenvecs, dtype=dtype),
    }


low_rank_options = {
    "large": get_low_rank_large_eigenvalues,
    "small": get_low_rank_small_eigenvalues,
}


def get_low_rank(
    mv: Callable,
    size: int,
    key: KeyType,
    maxiter: int = 20,
    local_dtype_switch: bool = True,
    eigval_mode: str = "small",
    dtype: DType = jnp.float64,
) -> dict:
    """Handling low rank calculations."""
    # Set local dtype if necessary
    if local_dtype_switch and dtype != jnp.float64:
        jax.config.update("jax_enable_x64", True)
        local_dtype = jnp.float64
    else:
        local_dtype = dtype

    # Get low rank
    low_rank_tuple = low_rank_options[eigval_mode](
        mv, maxiter=maxiter, size=size, key=key, dtype=local_dtype
    )

    # Adjust dtype and set global dtype variable.
    if dtype != local_dtype:
        low_rank_tuple = jax.tree.map(dtype, low_rank_tuple)

        if local_dtype_switch:
            jax.config.update("jax_enable_x64", False)

    return low_rank_tuple


# -------------------------------------------------------------
# Low rank mv callables
# -------------------------------------------------------------


def low_rank_mv_factory(low_rank_terms: dict):
    # Extract terms
    U = low_rank_terms["U"]
    S = low_rank_terms["S"]

    # Define mv-product
    def low_rank_mv(vec):
        return U @ (S[:, None] * (U.T @ vec))

    return low_rank_mv


def low_rank_plus_diagonal_mv_factory(low_rank_terms: dict):
    # Extract terms
    U = low_rank_terms["U"]
    S = low_rank_terms["S"]
    scalar = low_rank_terms["scalar"]

    # Define mv-product
    def low_rank_mv(vec):
        return scalar * vec + U @ (S[:, None] * (U.T @ vec))

    return low_rank_mv


def invert_low_rank_plus_diagonal(low_rank_terms: dict):
    # Extract terms
    U = low_rank_terms["U"]
    S = low_rank_terms["S"]
    scalar = low_rank_terms["scalar"]

    return {"U": U, "S": -S / (scalar * (S + scalar)), "scalar": scalar}


def inv_low_rank_plus_diagonal_mv_factory(low_rank_terms: dict):
    # inverse low rank terms
    low_rank_terms_inv = invert_low_rank_plus_diagonal(low_rank_terms)

    # create mv product
    return low_rank_mv_factory(low_rank_terms_inv)
