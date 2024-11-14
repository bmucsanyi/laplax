"""Posterior covariance functions for various curvature estimates."""

import jax
import jax.numpy as jnp

from laplax.curv.low_rank import get_low_rank, inv_low_rank_plus_diagonal_mv_factory
from laplax.types import Callable, PyTree
from laplax.util.flatten import (
    create_partial_pytree_flattener,
    create_pytree_flattener,
    wrap_function,
)
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
        msg = "Matrix is not positive definite"
        raise ValueError(msg)

    L_inv = jnp.transpose(jnp.flip(Lf, axis=(-2, -1)), axes=(-2, -1))
    Id = jnp.eye(prec.shape[-1], dtype=prec.dtype)
    L = jax.scipy.linalg.solve_triangular(L_inv, Id, trans="T")
    return L


def full_to_scale(prec: jax.Array) -> jax.Array:
    scale = prec_to_scale(prec)

    def scale_mv(vec):
        return scale @ vec

    return scale_mv, scale


def full_scale_to_cov(scale: jax.Array) -> jax.Array:
    full = scale @ scale.T

    def cov_mv(vec: jax.Array):
        return full @ vec

    return cov_mv


def full_with_prior(curv_est: jax.Array, **kwargs):
    return curv_est + kwargs.get("prior_prec") * jnp.eye(curv_est.shape[-1])


def create_full_curvature(mv: Callable, tree: PyTree):
    """Generate a full curvature approximation."""
    curv_est = todense(mv, like=tree)
    flatten_partial_tree, _ = create_partial_pytree_flattener(curv_est)
    return flatten_partial_tree(curv_est)


# ---------------------------------------------------------------------------------
# Diagonal
# ---------------------------------------------------------------------------------


def diag_with_prior(curv_est: jax.Array, **kwargs):
    return curv_est + kwargs.get("prior_prec") * jnp.ones_like(curv_est.shape[-1])


def diag_to_scale(arr: jax.Array):
    diag_inv = jnp.reciprocal(arr)

    def diag_mv(vec):
        return diag_inv * vec

    return diag_mv, diag_inv


def diag_scale_to_cov(arr: jax.Array):
    arr_sq = arr**2

    def diag_mv(vec):
        return arr_sq * vec

    return diag_mv


def create_diagonal_curvature(mv: Callable, **kwargs):
    """Generate a diagonal curvature."""
    curv_diagonal = diagonal(mv, tree=kwargs.get("tree"))
    return curv_diagonal


def create_low_rank_curvature(mv: Callable, size: int, **kwargs):
    """Generate a lcreate_pytree_flattener, ow-rank curvature approximations."""
    low_rank_terms = get_low_rank(mv, size, **kwargs)
    return low_rank_terms


CURVATURE_METHODS = {
    "full": create_full_curvature,
    "diagonal": create_diagonal_curvature,
    "low_rank": create_low_rank_curvature,
}

CURVATURE_PRIOR_METHODS = {
    "full": full_with_prior,
    "diagonal": diag_with_prior,
}

CURVATURE_INVERSE_METHODS = {
    "full": full_to_scale,
    "diagonal": diag_to_scale,
}

CURVATURE_COV_METHODS = {
    "full": full_scale_to_cov,
    "diagonal": diag_scale_to_cov,
}


def create_posterior_function(curvature_type: str, mv: Callable, **kwargs) -> Callable:
    """Factory function to create posterior covariance functions based on curvature type.

    Parameters:
        curvature_type: Type of curvature approximation ('full', 'diagonal', 'low_rank').
        mv: Function representing the curvature.
        **kwargs,
    """
    # Get general variables
    tree = kwargs.get("tree", None)
    if tree is None:
        msg = "Tree structure is required for full covariance."
        raise ValueError(msg)

    # Get terms
    curv_est = CURVATURE_METHODS[curvature_type](mv, **kwargs)
    flatten, unflatten = create_pytree_flattener(tree)

    def posterior_function(**kwargs):
        prec = CURVATURE_PRIOR_METHODS[curvature_type](curv_est=curv_est, **kwargs)
        scale, state = CURVATURE_INVERSE_METHODS[curvature_type](prec)
        cov = CURVATURE_COV_METHODS[curvature_type](state)
        return {
            "cov_mv": wrap_function(fn=cov, input_fn=flatten, output_fn=unflatten),
            "scale_mv": wrap_function(fn=scale, input_fn=flatten, output_fn=unflatten)
            if kwargs.get("return_scale", False)
            else None,
        }

    return posterior_function


# def create_full_cov(mv: Callable, tree: PyTree):
#     """Create posterior covariance function for full curvature estimate.

#     For inverting we flatten the PyTree and apply unflattening after.
#     """
#     # Get dense curvature estimate
#     curv_est = todense(mv, like=tree)  # Switched from todense to todensetree
#     flatten_partial_tree, _ = create_partial_pytree_flattener(curv_est)
#     flatten, unflatten = create_pytree_flattener(tree)
#     curv_est = flatten_partial_tree(curv_est)

#     def get_posterior(prior_prec, return_scale=False):
#         prec = curv_est + prior_prec * jnp.eye(curv_est.shape[-1])
#         scale = prec_to_scale(prec)
#         return {
#             "cov_mv": array_to_mv(scale, flatten, unflatten),
#             "scale_mv": array_to_mv(scale, flatten, unflatten)
#             if return_scale
#             else None,
#         }

#     return get_posterior


# # -----------------------------------------------------------------------
# # DIAGONAL
# # -----------------------------------------------------------------------


# def create_diagonal_cov(mv: Callable, size: int):
#     # Get diagonal curvature estimate
#     curv_diagonal = diagonal(mv, size=size)

#     def get_posterior(prior_prec):
#         prec = curv_diagonal + prior_prec
#         scale = jnp.invert(prec)
#         return array_to_mv(scale)

#     return get_posterior


# # -----------------------------------------------------------------------
# # LOW RANK
# # -----------------------------------------------------------------------


# def create_low_rank_cov(mv: Callable, size: int, **kwargs):
#     # Get low rank terms
#     low_rank_terms = get_low_rank(mv, size, **kwargs)

#     def get_posterior(prior_prec):
#         low_rank_terms["scalar"] = prior_prec
#         return inv_low_rank_plus_diagonal_mv_factory(low_rank_terms)

#     return get_posterior
