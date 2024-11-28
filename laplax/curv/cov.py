"""Posterior covariance functions for various curvature estimates."""

import jax
import jax.numpy as jnp

from laplax import util
from laplax.curv.low_rank import get_low_rank
from laplax.types import Callable, PyTree
from laplax.util.flatten import (
    create_partial_pytree_flattener,
    create_pytree_flattener,
    wrap_factory,
    wrap_function,
)
from laplax.util.mv import diagonal, todense

# -----------------------------------------------------------------------
# FULL
# -----------------------------------------------------------------------


def create_full_curvature(mv: Callable, tree: PyTree, **kwargs):
    """Generate a full curvature approximation."""
    del kwargs
    curv_est = todense(mv, like=tree)
    flatten_partial_tree, _ = create_partial_pytree_flattener(curv_est)
    return flatten_partial_tree(curv_est)


def full_with_prior(curv_est: jax.Array, **kwargs):
    return curv_est + kwargs.get("prior_prec") * jnp.eye(curv_est.shape[-1])


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


def full_prec_to_state(prec: jax.Array) -> jax.Array:
    scale = prec_to_scale(prec)

    return {"scale": scale}


def full_state_to_scale(state: dict) -> jax.Array:
    def scale_mv(vec: jax.Array) -> jax.Array:
        return state["scale"] @ vec

    return scale_mv


def full_state_to_cov(state: dict) -> jax.Array:
    cov = state["scale"] @ state["scale"].T

    def cov_mv(vec: jax.Array) -> jax.Array:
        return cov @ vec

    return cov_mv


# ---------------------------------------------------------------------------------
# Diagonal
# ---------------------------------------------------------------------------------


def create_diagonal_curvature(mv: Callable, **kwargs):
    """Generate a diagonal curvature."""
    curv_diagonal = diagonal(mv, tree=kwargs.get("tree"))
    return curv_diagonal


def diag_with_prior(curv_est: jax.Array, **kwargs):
    return curv_est + kwargs.get("prior_prec") * jnp.ones_like(curv_est.shape[-1])


def diag_prec_to_state(prec: jax.Array) -> dict:
    return {"scale": jnp.sqrt(jnp.reciprocal(prec))}


def diag_state_to_scale(state: dict) -> Callable:
    def diag_mv(vec):
        return state["scale"] * vec

    return diag_mv


def diag_state_to_cov(state: dict) -> Callable:
    arr = state["scale"] ** 2

    def diag_mv(vec):
        return arr * vec

    return diag_mv


# ---------------------------------------------------------------------------------
# Low-rank
# ---------------------------------------------------------------------------------


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


def create_low_rank_mv(low_rank_terms: dict) -> Callable:
    """Create a low-rank matrix-vector product."""
    U = low_rank_terms["U"]
    S = low_rank_terms["S"]
    scalar = low_rank_terms["scalar"]

    def low_rank_mv(vec: jax.Array) -> jax.Array:
        return scalar * vec + U @ (S * (U.T @ vec))

    return low_rank_mv


def low_rank_square(state: dict) -> Callable:
    scalar, eigvals = state["scalar"], state["S"]
    scalar_sq = scalar**2
    return {
        "U": state["U"],
        "S": (eigvals + scalar) ** 2 - scalar_sq,
        "scalar": scalar_sq,
    }


def low_rank_with_prior(curv_est: dict, **kwargs):
    curv_est["scalar"] = kwargs.get("prior_prec")
    return curv_est


def low_rank_prec_to_state(curv_est: dict):
    scalar = curv_est["scalar"]
    scalar_sqrt_inv = jnp.reciprocal(jnp.sqrt(scalar))
    eigvals = curv_est["S"]
    return {
        "scale": {
            "U": curv_est["U"],
            "S": jnp.reciprocal(jnp.sqrt(eigvals + scalar)) - scalar_sqrt_inv,
            "scalar": scalar_sqrt_inv,
        }
    }


def low_rank_state_to_scale(state: dict) -> Callable:
    return create_low_rank_mv(state["scale"])


def low_rank_state_to_cov(state: dict) -> Callable:
    return create_low_rank_mv(low_rank_square(state["scale"]))


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

CURVATURE_TO_POSTERIOR_STATE = {
    "full": full_prec_to_state,
    "diagonal": diag_prec_to_state,
    "low_rank": low_rank_prec_to_state,
}

CURVATURE_STATE_TO_SCALE = {
    "full": full_state_to_scale,
    "diagonal": diag_state_to_scale,
    "low_rank": low_rank_state_to_scale,
}

CURVATURE_STATE_TO_COV = {
    "full": full_state_to_cov,
    "diagonal": diag_state_to_cov,
    "low_rank": low_rank_state_to_cov,
}


def create_posterior_function(
    curvature_type: str, mv: Callable, structure: int | PyTree | None = None, **kwargs
) -> Callable:
    """Factory function to create posterior covariance functions based on curv. type.

    Parameters:
        curvature_type (str): Type of curvature approximation ('full', 'diagonal',
            'low_rank').
        mv (Callable): Function representing the curvature.
        **kwargs: Additional parameters required for specific curvature methods,
            including:
            - structure_format (Union[int, None]): Defines the format of the structure
                for matrix-vector products. If None or an integer, no
                flattening/unflattening is used.

    Returns:
        Callable: A posterior function that calculates posterior covariance and scale
            functions.
    """
    if structure is not None and not isinstance(structure, int | PyTree):
        msg = "structure must be an integer, a tuple, PyTree or None."
        raise ValueError(msg)

    # Create functions for flattening and unflattening if required
    if structure is None or isinstance(structure, int):
        flatten = unflatten = None
    else:
        # Use custom flatten/unflatten functions for complex pytrees
        flatten, unflatten = create_pytree_flattener(structure)

    # Retrieve the curvature estimator based on the provided type
    curv_estimator = CURVATURE_METHODS[curvature_type](mv, **kwargs)

    def posterior_function(**posterior_kwargs) -> dict[str, any]:
        """Posterior function to compute covariance and scale-related functions.

        Parameters:
            **posterior_kwargs: Additional arguments required for posterior
                computations.

        Returns:
            Dict[str, any]: Dictionary containing:
                - 'state': Updated state of the posterior.
                - 'cov_mv': Function to compute covariance matrix-vector product.
                - 'scale_mv': Function to compute scale matrix-vector product.
        """
        # Calculate posterior precision.
        precision = CURVATURE_PRIOR_METHODS[curvature_type](
            curv_est=curv_estimator, **posterior_kwargs
        )

        # Calculate posterior state
        state = CURVATURE_TO_POSTERIOR_STATE[curvature_type](precision)

        # Extract matrix-vector product
        scale_mv_from_state = CURVATURE_STATE_TO_SCALE[curvature_type]
        cov_mv_from_state = CURVATURE_STATE_TO_COV[curvature_type]

        return {
            "state": state,
            "cov_mv": wrap_factory(cov_mv_from_state, flatten, unflatten),
            "scale_mv": wrap_factory(scale_mv_from_state, flatten, unflatten),
        }

    return posterior_function
