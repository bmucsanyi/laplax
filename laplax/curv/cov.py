"""Posterior covariance functions for various curvature estimates."""

import jax
import jax.numpy as jnp

from laplax.curv.low_rank import get_low_rank, inv_low_rank_plus_diagonal_mv_factory
from laplax.types import Callable, PyTree
from laplax.util.flatten import create_partial_pytree_flattener
from laplax.util.mv import array_to_mv, diagonal, todense

# -----------------------------------------------------------------------
# FULL
# -----------------------------------------------------------------------


def prec_to_scale(prec: jax.Array) -> jax.Array:
    """Implementation of the corresponding torch function.

    See: torch.distributions.multivariate_normal._precision_to_scale_tril.
    """
    Lf = jnp.linalg.cholesky(jnp.flip(prec, axis=(-2, -1)))

    if jnp.any(jnp.isnan(Lf)):
        msg = "matrix is not positive definite"
        raise ValueError(msg)

    L_inv = jnp.transpose(jnp.flip(Lf, axis=(-2, -1)), axes=(-2, -1))
    Id = jnp.eye(prec.shape[-1], dtype=prec.dtype)
    L = jax.scipy.linalg.solve_triangular(L_inv, Id, trans="T")
    return L


def create_full_cov(mv: Callable, tree: PyTree):
    """Create posterior covariance function for full curvature estimate.

    For inverting we flatten the PyTree and apply unflattening after.
    """
    # Get dense curvature estimate
    curv_est = todense(mv, like=tree)
    flatten, unflatten = create_partial_pytree_flattener(curv_est)
    curv_est = flatten(curv_est)

    def get_posterior(prior_prec, return_scale=False):
        prec = curv_est + prior_prec * jnp.eye(curv_est.shape[-1])
        scale = prec_to_scale(prec)
        if return_scale:
            return array_to_mv(unflatten(scale))
        return array_to_mv(unflatten(scale @ scale.T))

    return get_posterior


# -----------------------------------------------------------------------
# DIAGONAL
# -----------------------------------------------------------------------


def create_diagonal_cov(mv: Callable, size: int):
    # Get diagonal curvature estimate
    curv_diagonal = diagonal(mv, size=size)

    def get_posterior(prior_prec):
        prec = curv_diagonal + prior_prec
        scale = jnp.invert(prec)
        return array_to_mv(scale)

    return get_posterior


# -----------------------------------------------------------------------
# LOW RANK
# -----------------------------------------------------------------------


def create_low_rank_cov(mv: Callable, size: int, **kwargs):
    # Get low rank terms
    low_rank_terms = get_low_rank(mv, size, **kwargs)

    def get_posterior(prior_prec):
        low_rank_terms["scalar"] = prior_prec
        return inv_low_rank_plus_diagonal_mv_factory(low_rank_terms)

    return get_posterior
