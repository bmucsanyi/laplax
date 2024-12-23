"""Posterior covariance functions for various curvature estimates."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from laplax import util
from laplax.curv.low_rank import LowRankTerms, get_low_rank_approximation
from laplax.enums import CurvApprox
from laplax.types import (
    Any,
    Array,
    Callable,
    CurvatureMV,
    FlatParams,
    Layout,
    Num,
    PosteriorState,
    PriorArguments,
    PyTree,
)
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


def create_full_curvature(
    mv: CurvatureMV, layout: Layout, **kwargs
) -> Num[Array, "P P"]:
    """Generate a full curvature approximation."""
    del kwargs
    curv_est = todense(mv, layout=layout)
    flatten_partial_tree, _ = create_partial_pytree_flattener(curv_est)
    return flatten_partial_tree(curv_est)


def full_with_prior(
    curv_est: Num[Array, "P P"],
    prior_arguments: PriorArguments,
) -> Num[Array, "P P"]:
    prior_prec = prior_arguments["prior_prec"]
    return curv_est + prior_prec * jnp.eye(curv_est.shape[-1])


def prec_to_scale(prec: Num[Array, "P P"]) -> Num[Array, "P P"]:
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


def full_prec_to_state(
    prec: Num[Array, "P P"],
) -> dict[str, Num[Array, "P P"]]:
    scale = prec_to_scale(prec)

    return {"scale": scale}


def full_state_to_scale(
    state: dict[str, Num[Array, "P P"]],
) -> Callable[[FlatParams], FlatParams]:
    def scale_mv(vec: FlatParams) -> FlatParams:
        return state["scale"] @ vec

    return scale_mv


def full_state_to_cov(
    state: dict[str, Num[Array, "P P"]],
) -> Callable[[FlatParams], FlatParams]:
    cov = state["scale"] @ state["scale"].T

    def cov_mv(vec: FlatParams) -> FlatParams:
        return cov @ vec

    return cov_mv


# ---------------------------------------------------------------------------------
# Diagonal
# ---------------------------------------------------------------------------------


def create_diagonal_curvature(mv: CurvatureMV, layout: Layout, **kwargs) -> FlatParams:
    """Generate a diagonal curvature."""
    del kwargs
    curv_diagonal = diagonal(mv, layout=layout)
    return curv_diagonal


def diag_with_prior(
    curv_est: FlatParams, prior_arguments: PriorArguments
) -> FlatParams:
    prior_prec = prior_arguments["prior_prec"]
    return curv_est + prior_prec * jnp.ones_like(curv_est.shape[-1])


def diag_prec_to_state(prec: FlatParams) -> dict[str, FlatParams]:
    return {"scale": jnp.sqrt(jnp.reciprocal(prec))}


def diag_state_to_scale(
    state: dict[str, FlatParams],
) -> Callable[[FlatParams], FlatParams]:
    def diag_mv(vec: FlatParams) -> FlatParams:
        return state["scale"] * vec

    return diag_mv


def diag_state_to_cov(
    state: dict[str, FlatParams],
) -> Callable[[FlatParams], FlatParams]:
    arr = state["scale"] ** 2

    def diag_mv(vec: FlatParams) -> FlatParams:
        return arr * vec

    return diag_mv


# ---------------------------------------------------------------------------------
# Low-rank
# ---------------------------------------------------------------------------------


def create_low_rank_curvature(
    mv: CurvatureMV, layout: Layout, **kwargs
) -> LowRankTerms:
    """Generate a create_pytree_flattener, low-rank curvature approximations."""
    flatten, unflatten = create_pytree_flattener(layout)
    nparams = util.tree.get_size(layout)
    mv = jax.vmap(
        wrap_function(fn=mv, input_fn=unflatten, output_fn=flatten),
        in_axes=-1,
        out_axes=-1,
    )  # Turn mv into matmul structure.
    low_rank_terms = get_low_rank_approximation(mv, size=nparams, **kwargs)

    return low_rank_terms


def create_low_rank_mv(
    low_rank_terms: LowRankTerms,
) -> Callable[[FlatParams], FlatParams]:
    """Create a low-rank matrix-vector product."""
    U, S, scalar = jax.tree_util.tree_leaves(low_rank_terms)

    def low_rank_mv(vec: FlatParams) -> FlatParams:
        return scalar * vec + U @ (S * (U.T @ vec))

    return low_rank_mv


def low_rank_square(state: LowRankTerms) -> LowRankTerms:
    U, S, scalar = jax.tree_util.tree_leaves(state)
    scalar_sq = scalar**2
    return LowRankTerms(
        U=U,
        S=(S + scalar) ** 2 - scalar_sq,
        scalar=scalar_sq,
    )


def low_rank_with_prior(
    curv_est: LowRankTerms, prior_arguments: PriorArguments
) -> LowRankTerms:
    prior_prec = prior_arguments["prior_prec"]
    curv_est.scalar = prior_prec
    return curv_est


def low_rank_prec_to_state(curv_est: LowRankTerms) -> dict[str, LowRankTerms]:
    U, S, scalar = jax.tree_util.tree_leaves(curv_est)
    scalar_sqrt_inv = jnp.reciprocal(jnp.sqrt(scalar))
    return {
        "scale": LowRankTerms(
            U=U,
            S=jnp.reciprocal(jnp.sqrt(S + scalar)) - scalar_sqrt_inv,
            scalar=scalar_sqrt_inv,
        )
    }


def low_rank_state_to_scale(
    state: dict[str, LowRankTerms],
) -> Callable[[FlatParams], FlatParams]:
    return create_low_rank_mv(state["scale"])


def low_rank_state_to_cov(
    state: dict[str, LowRankTerms],
) -> Callable[[FlatParams], FlatParams]:
    return create_low_rank_mv(low_rank_square(state["scale"]))


# ---------------------------------------------------------------------------------
# General api
# ---------------------------------------------------------------------------------

CurvatureKeyType = CurvApprox | str | None

CURVATURE_METHODS: dict[CurvatureKeyType, Callable] = {
    CurvApprox.FULL: create_full_curvature,
    CurvApprox.DIAGONAL: create_diagonal_curvature,
    CurvApprox.LOW_RANK: create_low_rank_curvature,
}

CURVATURE_PRIOR_METHODS: dict[CurvatureKeyType, Callable] = {
    CurvApprox.FULL: full_with_prior,
    CurvApprox.DIAGONAL: diag_with_prior,
    CurvApprox.LOW_RANK: low_rank_with_prior,
}

CURVATURE_TO_POSTERIOR_STATE: dict[CurvatureKeyType, Callable] = {
    CurvApprox.FULL: full_prec_to_state,
    CurvApprox.DIAGONAL: diag_prec_to_state,
    CurvApprox.LOW_RANK: low_rank_prec_to_state,
}

CURVATURE_STATE_TO_SCALE: dict[CurvatureKeyType, Callable] = {
    CurvApprox.FULL: full_state_to_scale,
    CurvApprox.DIAGONAL: diag_state_to_scale,
    CurvApprox.LOW_RANK: low_rank_state_to_scale,
}

CURVATURE_STATE_TO_COV: dict[CurvatureKeyType, Callable] = {
    CurvApprox.FULL: full_state_to_cov,
    CurvApprox.DIAGONAL: diag_state_to_cov,
    CurvApprox.LOW_RANK: low_rank_state_to_cov,
}


@dataclass
class Posterior:
    state: PosteriorState
    cov_mv: Callable[[PosteriorState], Callable[[FlatParams], FlatParams]]
    scale_mv: Callable[[PosteriorState], Callable[[FlatParams], FlatParams]]


def create_posterior_function(
    curvature_type: CurvApprox | str,
    mv: CurvatureMV,
    layout: Layout | None = None,
    **kwargs,
) -> Callable:
    """Factory function to create posterior covariance functions based on curv. type.

    Parameters:
        curvature_type (str): Type of curvature approximation ('full', 'diagonal',
            'low_rank').
        mv (Callable): Function representing the curvature.
        **kwargs: Additional parameters required for specific curvature methods,
            including:
            - layout (Union[int, None]): Defines the format of the layout
                for matrix-vector products. If None or an integer, no
                flattening/unflattening is used.

    Returns:
        Callable: A posterior function that calculates posterior covariance and scale
            functions.
    """
    if layout is not None and not isinstance(layout, int | PyTree):
        msg = "Layout must be an integer, PyTree or None."
        raise ValueError(msg)

    # Create functions for flattening and unflattening if required
    if layout is None or isinstance(layout, int):
        flatten = unflatten = None
    else:
        # Use custom flatten/unflatten functions for complex pytrees
        flatten, unflatten = create_pytree_flattener(layout)

    # Retrieve the curvature estimator based on the provided type
    curv_estimator = CURVATURE_METHODS[curvature_type](mv, layout=layout, **kwargs)

    def posterior_function(prior_arguments: PriorArguments) -> PosteriorState:
        """Posterior function to compute covariance and scale-related functions.

        Parameters:
            prior_arguments (PriorArguments): Prior arguments for the posterior.

        Returns:
            PosteriorState: Dictionary containing:
                - 'state': Updated state of the posterior.
                - 'cov_mv': Function to compute covariance matrix-vector product.
                - 'scale_mv': Function to compute scale matrix-vector product.
        """
        # Calculate posterior precision.
        precision = CURVATURE_PRIOR_METHODS[curvature_type](
            curv_est=curv_estimator, prior_arguments=prior_arguments
        )

        # Calculate posterior state
        state = CURVATURE_TO_POSTERIOR_STATE[curvature_type](precision)

        # Extract matrix-vector product
        scale_mv_from_state = CURVATURE_STATE_TO_SCALE[curvature_type]
        cov_mv_from_state = CURVATURE_STATE_TO_COV[curvature_type]

        return Posterior(
            state=state,
            cov_mv=wrap_factory(cov_mv_from_state, flatten, unflatten),
            scale_mv=wrap_factory(scale_mv_from_state, flatten, unflatten),
        )

    return posterior_function


# ----------------------------------------------------------------------------------
# Register new curvature methods
# ----------------------------------------------------------------------------------


def register_curvature_method(
    name: str,
    *,
    create_fn: Callable[[CurvatureMV, Layout, Any], Any] | None = None,
    prior_fn: Callable | None = None,
    posterior_fn: Callable | None = None,
    scale_fn: Callable[[PosteriorState], Callable[[FlatParams], FlatParams]]
    | None = None,
    cov_fn: Callable[[PosteriorState], Callable[[FlatParams], FlatParams]]
    | None = None,
    default: CurvApprox | None = None,
) -> None:
    """Register a new curvature method with optional custom functions.

    Parameters:
        name (str): Name of the new curvature method.
        create_fn (Callable, optional): Custom curvature creation function.
        prior_fn (Callable, optional): Custom prior function.
        posterior_fn (Callable, optional): Custom posterior state function.
        scale_fn (Callable, optional): Custom state-to-scale function.
        cov_fn (Callable, optional): Custom state-to-cov function.
        default (CurvApprox, optional): Default method to inherit from if custom
            functions are not provided.

    Raises:
        ValueError: If neither a default method is provided nor all required
            functions are specified.
    """
    # Check whether default is given
    if default is None and not all((
        create_fn,
        prior_fn,
        posterior_fn,
        scale_fn,
        cov_fn,
    )):
        missing_functions = [
            fn_name
            for fn_name, fn in zip(
                ["create_fn", "prior_fn", "posterior_fn", "scale_fn", "cov_fn"],
                [create_fn, prior_fn, posterior_fn, scale_fn, cov_fn],
                strict=True,
            )
            if fn is None
        ]
        msg = (
            "Either a default method must be provided or the following functions must "
            f"be specified: {', '.join(missing_functions)}."
        )
        raise ValueError(msg)

    CURVATURE_METHODS[name] = create_fn or CURVATURE_METHODS[default]
    CURVATURE_PRIOR_METHODS[name] = prior_fn or CURVATURE_PRIOR_METHODS[default]
    CURVATURE_TO_POSTERIOR_STATE[name] = (
        posterior_fn or CURVATURE_TO_POSTERIOR_STATE[default]
    )
    CURVATURE_STATE_TO_SCALE[name] = scale_fn or CURVATURE_STATE_TO_SCALE[default]
    CURVATURE_STATE_TO_COV[name] = cov_fn or CURVATURE_STATE_TO_COV[default]
