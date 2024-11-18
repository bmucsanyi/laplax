"""Posterior covariance functions for various curvature estimates."""

import jax
import jax.numpy as jnp

from laplax import util
from laplax.curv.low_rank import get_low_rank
from laplax.types import Callable, PyTree
from laplax.util.flatten import (
    create_partial_pytree_flattener,
    create_pytree_flattener,
    wrap_function,
)
from laplax.util.mv import diagonal, todense

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


def create_full_curvature(mv: Callable, tree: PyTree, **kwargs):
    """Generate a full curvature approximation."""
    del kwargs
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


# ---------------------------------------------------------------------------------
# Low-rank
# ---------------------------------------------------------------------------------


def create_low_rank_mv(low_rank_terms: dict) -> Callable:
    """Create a low-rank matrix-vector product."""
    U = low_rank_terms["U"]
    S = low_rank_terms["S"]
    scalar = low_rank_terms["scalar"]

    def low_rank_mv(vec: jax.Array) -> jax.Array:
        return scalar * vec + U @ (S * (U.T @ vec))

    return low_rank_mv


def low_rank_with_prior(curv_est: dict, **kwargs):
    curv_est["scalar"] = kwargs.get("prior_prec")
    return curv_est


def low_rank_to_scale(curv_est: dict):
    scalar = curv_est["scalar"]
    scalar_sqrt_inv = jnp.reciprocal(jnp.sqrt(scalar))
    eigvals = curv_est["S"]
    new_curv_est = {
        "U": curv_est["U"],
        "S": jnp.reciprocal(jnp.sqrt(eigvals + scalar)) - scalar_sqrt_inv,
        "scalar": scalar_sqrt_inv,
    }
    return create_low_rank_mv(new_curv_est), new_curv_est


def low_rank_scale_to_cov(curv_est: dict):
    scalar = curv_est["scalar"]
    scalar_sq = scalar**2
    eigvals = curv_est["S"]
    new_curv_est = {
        "U": curv_est["U"],
        "S": (eigvals + scalar) ** 2 - scalar_sq,
        "scalar": scalar_sq,
    }
    return create_low_rank_mv(new_curv_est)


def create_low_rank_curvature(mv: Callable, **kwargs):
    """Generate a lcreate_pytree_flattener, ow-rank curvature approximations."""
    tree = kwargs.get("tree")
    flatten, unflatten = create_pytree_flattener(tree)
    nparams = util.tree.get_size(tree)
    mv = jax.vmap(
        wrap_function(fn=mv, input_fn=unflatten, output_fn=flatten),
        in_axes=-1,
        out_axes=-1,
    )  # Needs matmul structure.
    low_rank_terms = get_low_rank(mv, size=nparams, **kwargs)

    return low_rank_terms


# ---------------------------------------------------------------------------------
# General api
# ---------------------------------------------------------------------------------

CURVATURE_METHODS = {
    "full": create_full_curvature,
    "diagonal": create_diagonal_curvature,
    "low_rank": create_low_rank_curvature,
}

CURVATURE_PRIOR_METHODS = {
    "full": full_with_prior,
    "diagonal": diag_with_prior,
    "low_rank": low_rank_with_prior,
}

CURVATURE_INVERSE_METHODS = {
    "full": full_to_scale,
    "diagonal": diag_to_scale,
    "low_rank": low_rank_to_scale,
}

CURVATURE_COV_METHODS = {
    "full": full_scale_to_cov,
    "diagonal": diag_scale_to_cov,
    "low_rank": low_rank_scale_to_cov,
}


def create_posterior_function(curvature_type: str, mv: Callable, **kwargs) -> Callable:
    """Factory for creating posterior covariance functions based on curvature type.

    Parameters:
        curvature_type: Type of curvature approx. ('full', 'diagonal', 'low_rank').
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
