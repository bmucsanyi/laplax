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
    """Generate a full curvature approximation.

    The curvature is densed and flattened into a 2D array, that corresponds to the
    flattened parameter layout.

    Args:
        mv: Matrix-vector product function representing the curvature.
        layout: Structure defining the parameter layout that is assumed by the
            matrix-vector product function.
        **kwargs: Additional arguments (unused).

    Returns:
        A dense matrix representing the full curvature approximation.
    """
    del kwargs
    curv_est = todense(mv, layout=layout)
    flatten_partial_tree, _ = create_partial_pytree_flattener(curv_est)
    return flatten_partial_tree(curv_est)


def full_with_prior(
    curv_est: Num[Array, "P P"],
    prior_arguments: PriorArguments,
) -> Num[Array, "P P"]:
    """Add prior precision to the curvature estimate.

    The prior precision (of an isotropic Gaussian prior) is read of the prior_arguments
    dictionary and added to the curvature estimate.

    Args:
        curv_est: Full curvature estimate matrix.
        prior_arguments: Dictionary containing prior precision as 'prior_prec'.

    Returns:
        Updated curvature matrix with added prior precision.
    """
    prior_prec = prior_arguments["prior_prec"]
    return curv_est + prior_prec * jnp.eye(curv_est.shape[-1])


def prec_to_scale(prec: Num[Array, "P P"]) -> Num[Array, "P P"]:
    """Convert precision matrix to scale matrix using Cholesky decomposition.

    Implementation of the corresponding torch function for converting a precision
    matrix to a scale lower triangular matrix.
    See: torch.distributions.multivariate_normal._precision_to_scale_tril.

    Args:
        prec: Precision matrix to convert.

    Returns:
        Scale matrix L where L @ L.T is the covariance matrix.

    Raises:
        ValueError: If the precision matrix is not positive definite.
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
    """Convert precision matrix to scale matrix.

    The provided precision matrix is converted to a scale matrix, which is the lower
    triangular matrix L such that L @ L.T is the covariance matrix using
    `prec_to_scale`.

    Args:
        prec: Precision matrix to convert.

    Returns:
        Scale matrix L where L @ L.T is the covariance matrix.
    """
    scale = prec_to_scale(prec)

    return {"scale": scale}


def full_state_to_scale(
    state: dict[str, Num[Array, "P P"]],
) -> Callable[[FlatParams], FlatParams]:
    """Create a scale matrix-vector product function.

    The scale matrix is read from the state dictionary and is used to create a
    corresponding matrix-vector product function representing the action of the scale
    matrix on a vector.

    Args:
        state: Dictionary containing the scale matrix.

    Returns:
        A function that computes the scale matrix-vector product.
    """

    def scale_mv(vec: FlatParams) -> FlatParams:
        return state["scale"] @ vec

    return scale_mv


def full_state_to_cov(
    state: dict[str, Num[Array, "P P"]],
) -> Callable[[FlatParams], FlatParams]:
    """Create a covariance matrix-vector product function.

    The scale matrix is read from the state dictionary and is used to create a
    corresponding matrix-vector product function representing the action of the cov
    matrix on a vector. The covariance matrix is computed as the product of the scale
    matrix and its transpose.

    Args:
        state: Dictionary containing the scale matrix.

    Returns:
        A function that computes the covariance matrix-vector product.
    """
    cov = state["scale"] @ state["scale"].T

    def cov_mv(vec: FlatParams) -> FlatParams:
        return cov @ vec

    return cov_mv


# ---------------------------------------------------------------------------------
# Diagonal
# ---------------------------------------------------------------------------------


def create_diagonal_curvature(mv: CurvatureMV, layout: Layout, **kwargs) -> FlatParams:
    """Generate a diagonal curvature.

    The diagonal of the curvature matrix-vector product is computed as an approximation
    to the full matrix.

    Args:
        mv: Matrix-vector product function representing the curvature.
        layout: Structure defining the parameter layout that is assumed by the
            matrix-vector product function.
        **kwargs: Additional arguments (unused).

    Returns:
        A 1D array representing the diagonal curvature.
    """
    del kwargs
    curv_diagonal = diagonal(mv, layout=layout)
    return curv_diagonal


def diag_with_prior(
    curv_est: FlatParams, prior_arguments: PriorArguments
) -> FlatParams:
    """Add prior precision to the diagonal curvature estimate.

    The prior precision (of an isotropic Gaussian prior) is read of the prior_arguments
    dictionary and added to the diagonal curvature estimate.

    Args:
        curv_est: Diagonal curvature estimate.
        prior_arguments: Dictionary containing prior precision as 'prior_prec'.

    Returns:
        Updated diagonal curvature with added prior precision.
    """
    prior_prec = prior_arguments["prior_prec"]
    return curv_est + prior_prec * jnp.ones_like(curv_est.shape[-1])


def diag_prec_to_state(prec: FlatParams) -> dict[str, FlatParams]:
    """Convert precision matrix to scale matrix.

    The provided diagonal precision matrix is converted to the corresponding scale
    diagonal, which is returned as a PosteriorState dictionary.

    Args:
        prec: Precision matrix to convert.

    Returns:
        Scale matrix L where L @ L.T is the covariance matrix.
    """
    return {"scale": jnp.sqrt(jnp.reciprocal(prec))}


def diag_state_to_scale(
    state: dict[str, FlatParams],
) -> Callable[[FlatParams], FlatParams]:
    """Create a scale matrix-vector product function.

    The diagonal scale matrix is read from the state dictionary and is used to create
    a corresponding matrix-vector product function representing the action of the
    diagonal scale matrix on a vector.

    Args:
        state: Dictionary containing the diagonal scale matrix.

    Returns:
        A function that computes the diagonal scale matrix-vector product.
    """

    def diag_mv(vec: FlatParams) -> FlatParams:
        return state["scale"] * vec

    return diag_mv


def diag_state_to_cov(
    state: dict[str, FlatParams],
) -> Callable[[FlatParams], FlatParams]:
    """Create a covariance matrix-vector product function.

    The diagonal covariance matrix is computed as the product of the diagonal scale
    matrix with itself.

    Args:
        state: Dictionary containing the diagonal scale matrix.

    Returns:
        A function that computes the diagonal covariance matrix-vector product.
    """
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
    """Generate a low-rank curvature approximations.

    The low-rank curvature is computed as an approximation to the full curvature
    matrix using the provided matrix-vector product function and the LOBPCG algorithm.

    Args:
        mv: Matrix-vector product function representing the curvature.
        layout: Structure defining the parameter layout that is assumed by the
            matrix-vector product function.
        **kwargs: Additional arguments (unused).

    Returns:
        A LowRankTerms object representing the low-rank curvature approximation.
    """
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
    r"""Create a low-rank matrix-vector product function.

    The low-rank matrix-vector product is computed as the sum of the scalar multiple
    of the vector by the scalar and the product of the matrix-vector product of the
    eigenvectors and the eigenvalues times the eigenvector-vector product:

    $$
    scalar * \text{vec} + U @ (S * (U.T @ \text{vec}))
    $$

    Args:
        low_rank_terms: Low-rank curvature approximation.

    Returns:
        A function that computes the low-rank matrix-vector product.
    """
    U, S, scalar = jax.tree_util.tree_leaves(low_rank_terms)

    def low_rank_mv(vec: FlatParams) -> FlatParams:
        return scalar * vec + U @ (S * (U.T @ vec))

    return low_rank_mv


def low_rank_square(state: LowRankTerms) -> LowRankTerms:
    r"""Square the low-rank curvature approximation.

    This returns the LowRankTerms which correspond to the squared low rank
    approximation.

    $$ (U S U^{\top} + scalar I)**2
    = scalar**2 + U ((S + scalar) ** 2 - scalar**2) U^{\top} $$

    Args:
        state: Low-rank curvature approximation.

    Returns:
        A LowRankTerms object representing the squared low-rank curvature approximation.
    """
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
    """Add prior precision to the low-rank curvature estimate.

    The prior precision (of an isotropic Gaussian prior) is read from the
    `prior_arguments` dictionary and added to the scalar component of the
    LowRankTerms.

    Args:
        curv_est: Low-rank curvature approximation.
        prior_arguments: Dictionary containing prior precision
            as 'prior_prec'.

    Returns:
        LowRankTerms: Updated low-rank curvature approximation with added prior
            precision.
    """
    prior_prec = prior_arguments["prior_prec"]
    curv_est.scalar = prior_prec
    return curv_est


def low_rank_prec_to_state(curv_est: LowRankTerms) -> dict[str, LowRankTerms]:
    """Convert the low-rank precision representation to a posterior state.

    The scalar component and eigenvalues of the low-rank curvature estimate
    are transformed to represent the posterior scale, creating again a `LowRankTerms`
    representation.

    Args:
        curv_est: Low-rank curvature estimate.

    Returns:
        A dictionary with the posterior state represented as `LowRankTerms`.
    """
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
    """Create a matrix-vector product function for the scale matrix.

    The state dictionary containing the low-rank representation of the covariance state
    is used to create a function that computes the matrix-vector product for the scale
    matrix.

    Args:
        state: Dictionary containing the low-rank scale.

    Returns:
        A function that computes the scale matrix-vector product.
    """
    return create_low_rank_mv(state["scale"])


def low_rank_state_to_cov(
    state: dict[str, LowRankTerms],
) -> Callable[[FlatParams], FlatParams]:
    """Create a matrix-vector product function for the covariance matrix.

    The state dictionary containing the low-rank representation of the covariance state
    is used to create a function that computes the matrix-vector product for the
    covariance matrix.

    Args:
        state: Dictionary containing the low-rank scale.

    Returns:
        A function that computes the covariance matrix-vector product.
    """
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
    """Factory function to create the posterior function given a curvature type.

    This sets up the posterior_function which can then be initiated using
    prior_arguments by computing a specified curvature approximation and encoding the
    sequential computational order of CURVATURE_PRIOR_METHODS,
    CURVATURE_TO_POSTERIOR_STATE, CURVATURE_STATE_TO_SCALE, and CURVATURE_STATE_TO_COV.
    All methods are selected from the corresponding dictionary by the curvature_type
    argument. New methods can be registered using the register_curvature_method.

    Args:
        curvature_type: Type of curvature approximation ('full', 'diagonal',
            'low_rank').
        mv: Function representing the curvature.
        layout: Defines the format of the layout
                for matrix-vector products. If None or an integer, no
                flattening/unflattening is used.
        **kwargs: Additional key-word arguments are only passed to the curvature
            estimation function.

    Returns:
        Callable: A posterior function that takes the prior_arguments and returns the
            posterior_state.
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
            prior_arguments: Prior arguments for the posterior.

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

    This function allows adding new curvature methods with their corresponding
    functions for creating curvature estimates, adding prior information,
    computing posterior states, and deriving matrix-vector product functions
    for scale and covariance.

    Args:
        name: Name of the new curvature method.
        create_fn: Custom function to create the curvature
            estimate. Defaults to None.
        prior_fn: Custom function to incorporate prior
            information. Defaults to None.
        posterior_fn: Custom function to compute posterior
            states. Defaults to None.
        scale_fn: Custom function to compute scale
            matrix-vector products. Defaults to None.
        cov_fn: Custom function to compute covariance
            matrix-vector products. Defaults to None.
        default: Default method to inherit missing
            functionality from. Defaults to None.

    Raises:
        ValueError: If no default is provided and required functions are missing.
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
