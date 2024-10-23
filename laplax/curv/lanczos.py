"""Lanczos implementations for low rank approximations."""

import jax
import jax.numpy as jnp

from laplax.types import Callable, KeyType, ShapeType


def lanczos_random_init(key: KeyType, shape: ShapeType) -> jax.Array:
    return jax.random.normal(key, shape)


def lanczos_isqrt_full_reortho(
    A: Callable,
    b: jax.Array,
    maxiter: int | None = None,
    tol: float | None = None,
    *,
    overwrite_b: bool = False,
) -> jax.Array:
    """Inverse square root lanczos algorithm.

    Example inputs:
    >>> n = 100
    >>> A = jax.random.normal(jax.random.PRNGKey(0), (n, n))  # Example linear operator
    >>> b = jax.random.normal(jax.random.PRNGKey(1), (A.shape[0],), dtype=jnp.float32)
    >>> lanczos_isqrt_full_reortho(A, b, maxiter=100, tol=1e-6)
    """
    # Convert b to a JAX array and normalize
    # A = A.astype(jnp.float64)  # set to float64
    b = jnp.asarray(b, dtype=jnp.float64)  # set to float64
    b /= jnp.linalg.norm(b, 2)
    # Determine the combined dtype
    dtype = b.dtype

    # Set default maxiter
    if maxiter is None:
        maxiter = A.shape[1]  # Assuming A has attribute shape

    # Set the square tolerance
    sqtol = jnp.finfo(dtype).eps if tol is None else tol**2

    # Preallocate tensors for storing iterations
    ds = jnp.empty((b.size, maxiter), dtype=dtype)
    rs = jnp.empty((b.size, maxiter + 1), dtype=dtype)
    rs_norm_sq = jnp.zeros(maxiter + 1, dtype=dtype)

    # Initialize loop variables
    k = 0
    rs = rs.at[:, 0].set(b)
    rs_norm_sq = rs_norm_sq.at[0].set(1.0)  # Since b is normalized rs[:, 0] = b

    p = b.copy() if not overwrite_b else b

    # Set eta to an arbitrary large value
    eta = 9999.0
    op_mul = jax.jit(A)  # TODO(2bys): Check whether we should use jit outside.
    while k < maxiter and rs_norm_sq[k] > sqtol and eta > 1e-6:
        # Compute search direction
        if k > 0:
            tau = rs_norm_sq[k] / rs_norm_sq[k - 1]
            p = tau * p + rs[:, k]  # p_k = tau * p_{k-1} + rs[:, k]

        # Compute modified Lanczos vector
        w = jnp.asarray(op_mul(jnp.asarray(p, dtype=jnp.float32)), dtype=jnp.float64)
        eta = p @ w
        if eta > 1e-8:
            ds = ds.at[:, k].set(p / jnp.sqrt(eta))
            # Update residual
            mu = rs_norm_sq[k] / eta
            rs = rs.at[:, k + 1].set(rs[:, k] - mu * w)
            # Full reorthogonalization of residual (double Gram-Schmidt)
            proj = rs[:, :k].T @ rs[:, k + 1]
            rs = rs.at[:, k + 1].set(rs[:, k + 1] - rs[:, :k] @ (proj / rs_norm_sq[:k]))
            proj = rs[:, :k].T @ rs[:, k + 1]
            rs = rs.at[:, k + 1].set(rs[:, k + 1] - rs[:, :k] @ (proj / rs_norm_sq[:k]))
            rs_norm_sq = rs_norm_sq.at[k + 1].set(rs[:, k + 1] @ rs[:, k + 1])
            k += 1
    print(f"GGN low_rank approximation converged with rank {k}.")  # noqa: T201
    return ds[:, :k].astype(jnp.float32)
