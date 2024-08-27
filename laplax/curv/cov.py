# noqa: D100
# def get_cov_scale(prior): Callable.
import jax
import jax.numpy as jnp


def prec_to_scale(P: jax.Array) -> jax.Array:
    """Implementation of the corresponding torch function.

    See: torch.distributions.multivariate_normal._precision_to_scale_tril.
    """
    Lf = jnp.linalg.cholesky(jnp.flip(P, axis=(-2, -1)))

    if jnp.any(jnp.isnan(Lf)):
        msg = "matrix is not positive definite"
        raise ValueError(msg)

    L_inv = jnp.transpose(jnp.flip(Lf, axis=(-2, -1)), axes=(-2, -1))
    Id = jnp.eye(P.shape[-1], dtype=P.dtype)
    L = jax.scipy.linalg.solve_triangular(L_inv, Id, trans="T")
    return L
