"""Pushforward functions for weight space uncertainty.

This file contains the additional functions for pushing forward
weight space uncertainty onto output uncertainty.
"""

import math
from collections import OrderedDict

import jax
import jax.numpy as jnp

from laplax import util
from laplax.curv.cov import Posterior
from laplax.eval.utils import finalize_functions
from laplax.types import (
    Array,
    Callable,
    InputArray,
    KeyType,
    ModelFn,
    Params,
    PosteriorState,
    PredArray,
    PriorArguments,
    PyTree,
)
from laplax.util.ops import lmap, precompute_list

# -------------------------------------------------------------------------
# General utilities
# -------------------------------------------------------------------------


def pred_fn(**kwargs):
    return kwargs.get("pred")


def get_normal_weight_samples(
    key: KeyType,
    mean: PyTree,
    scale_mv: Callable[[PyTree], PyTree],
) -> PyTree:
    return util.tree.add(mean, scale_mv(util.tree.randn_like(key, mean)))


def set_get_weight_sample(key, mean, scale_mv, n_weight_samples, **kwargs):
    keys = jax.random.split(key, n_weight_samples)

    def get_weight_sample(idx):
        return get_normal_weight_samples(keys[idx], mean, scale_mv)

    return precompute_list(
        get_weight_sample,
        jnp.arange(n_weight_samples),
        option=kwargs.get("precompute_samples", "samples"),
    )


# -------------------------------------------------------------------------
# Monte Carlo pushforward
# -------------------------------------------------------------------------


def mc_pred_mean_fn(pred_ensemble: PredArray, **kwargs):
    del kwargs
    return util.tree.mean(pred_ensemble, axis=0)


def mc_pred_cov_fn(pred_ensemble: PredArray, **kwargs):
    del kwargs
    return util.tree.cov(
        pred_ensemble.reshape(pred_ensemble.shape[0], -1), rowvar=False
    )


def mc_pred_var_fn(pred_ensemble: PredArray, pred_cov: Array | None = None, **kwargs):
    """Compute variance of ensemble.

    Dependent compute:
    - (possible) pred_cov exists.
    - (fallback) pred_ensemble
    """
    del kwargs
    return (
        jnp.diagonal(pred_cov)
        if pred_cov is not None
        else util.tree.var(pred_ensemble, axis=0)
    )


def mc_pred_std_fn(pred_ensemble: PredArray, pred_var: Array | None = None, **kwargs):
    """Compute std of ensemble.

    Dependent compute:
    - (notimplementederror) pred_cov exists.
    - (possible) pred_var exists. (?)
    - (fallback) pred_ensemble
    """
    del kwargs
    return (
        jnp.sqrt(pred_var)
        if pred_var is not None
        else util.tree.std(pred_ensemble, axis=0)
    )


def mc_samples_fn(pred_ensemble: PredArray, n_samples: int = 5, **kwargs):
    """Select samples from ensemble."""
    del kwargs
    return util.tree.tree_slice(pred_ensemble, 0, n_samples)


DEFAULT_MC_FUNCTIONS = OrderedDict([
    ("pred", pred_fn),
    ("pred_mean", mc_pred_mean_fn),
    ("pred_var", mc_pred_var_fn),
    ("pred_std", mc_pred_std_fn),
    ("pred_cov", mc_pred_cov_fn),
    ("samples", mc_samples_fn),
])


def set_mc_pushforward(
    key: KeyType,
    model_fn: ModelFn,
    mean: Params,
    posterior: Callable[[PriorArguments], Posterior],
    prior_arguments: PriorArguments,
    n_weight_samples: int,
    pushforward_functions: OrderedDict = DEFAULT_MC_FUNCTIONS,
    **kwargs,
) -> Callable[[InputArray], dict[str, Array]]:
    # Create weight sample function
    posterior_state = posterior(**prior_arguments)
    scale_mv = posterior_state.scale_mv(posterior_state.state)

    get_weight_sample = set_get_weight_sample(
        key,
        mean,
        scale_mv,
        n_weight_samples,
    )

    # Create prob predictive function
    def prob_predictive(input: InputArray) -> dict[str, Array]:
        def compute_pred_ptw(idx: int) -> PredArray:
            weight_sample = get_weight_sample(idx)
            return model_fn(input=input, params=weight_sample)

        pred = model_fn(input=input, params=mean)
        pred_ensemble = lmap(
            compute_pred_ptw,
            jnp.arange(n_weight_samples),
            batch_size=kwargs.get("lmap_pred_ptw", "weight"),
        )

        return finalize_functions(
            functions=pushforward_functions,
            results={"pred": pred},
            pred_ensemble=pred_ensemble,
        )

    return prob_predictive


# -------------------------------------------------------------------------
# Linearized pushforward
# -------------------------------------------------------------------------


def lin_pred_var_fn(pred: PredArray, **kwargs):
    # Get argumentscurv_est.curv_est.
    cov = kwargs.get("pred_cov", kwargs.get("cov_mv"))
    del kwargs

    # Compute diagonal as variance
    var = util.mv.diagonal(cov, layout=math.prod(pred.shape))
    return var


def lin_pred_std_fn(**kwargs):
    var = kwargs.get("pred_var", lin_pred_var_fn(**kwargs))
    return util.tree.sqrt(var)


def lin_pred_cov_fn(
    pred: PredArray, cov_mv: Callable[[PredArray], PredArray], **kwargs
):
    del kwargs
    return util.mv.todense(cov_mv, layout=pred)


def lin_n_samples_fn(
    scale_mv: Callable[[PredArray], PredArray],
    weight_samples: Callable[[int], PredArray],
    n_samples: int = 5,
    **kwargs,
):
    return lmap(
        lambda i: scale_mv(weight_samples(i)),
        jnp.arange(n_samples),
        batch_size=kwargs.get("lmap_lin_samples", "weight"),
    )


DEFAULT_LIN_FINALIZE = OrderedDict([
    ("pred", pred_fn),
    ("pred_mean", pred_fn),
    ("pred_var", lin_pred_var_fn),
    ("pred_std", lin_pred_std_fn),
    ("pred_cov", lin_pred_cov_fn),
    ("samples", lin_n_samples_fn),
])


def set_output_cov_mv(
    posterior_state: Posterior,
    input: InputArray,
    jvp: Callable[[InputArray, Params], PredArray],
    vjp: Callable[[InputArray, PredArray], Params],
):
    cov_mv = posterior_state.cov_mv(posterior_state.state)
    scale_mv = posterior_state.scale_mv(posterior_state.state)

    def output_cov_mv(vec: PredArray) -> PredArray:
        return jvp(input, cov_mv(vjp(input, vec)[0]))

    def output_cov_scale_mv(vec: PredArray) -> PredArray:
        return jvp(input, scale_mv(vec))

    return {"cov_mv": output_cov_mv, "scale_mv": output_cov_scale_mv}


def set_lin_pushforward(
    key: KeyType,
    model_fn: ModelFn,
    mean: Params,
    posterior: Callable[[PriorArguments], PosteriorState],
    prior_arguments: PriorArguments,
    pushforward_functions: OrderedDict = DEFAULT_LIN_FINALIZE,
    **kwargs,
) -> Callable:
    # Create posterior state
    posterior_state = posterior(**prior_arguments)

    # Create pushforward functions
    def pf_jvp(input: InputArray, vector: Params) -> PredArray:
        return jax.jvp(
            lambda p: model_fn(input=input, params=p),
            (mean,),
            (vector,),
        )[1]

    def pf_vjp(input: InputArray, vector: PredArray) -> Params:
        out, vjp_fun = jax.vjp(lambda p: model_fn(input=input, params=p), mean)
        return vjp_fun(vector.reshape(out.shape))

    # Create scale mv
    if "samples" in pushforward_functions:
        n_samples = kwargs.pop("n_samples")
        get_weight_samples = set_get_weight_sample(
            key,
            mean,
            posterior_state.scale_mv(posterior_state.state),
            n_samples,
            **kwargs,
        )
    else:
        get_weight_samples = None
        n_samples = 0

    def prob_predictive(input: InputArray) -> dict[str, Array]:
        # Mean prediction
        pred = model_fn(input=input, params=mean)

        # Compute prediction
        return finalize_functions(
            functions=pushforward_functions,
            results={"pred": pred},
            **set_output_cov_mv(posterior_state, input=input, jvp=pf_jvp, vjp=pf_vjp),
            input=input,
            weight_samples=get_weight_samples,
            n_samples=n_samples,
            **kwargs,
        )

    return prob_predictive


# ------------------------------------------------------------------------
# Posterior GP kernel
# ------------------------------------------------------------------------


def set_posterior_gp_kernel(
    model_fn: ModelFn,
    mean: Params,
    posterior: Callable[[PriorArguments], PosteriorState],
    prior_arguments: PriorArguments,
    **kwargs,
) -> Callable:
    """Constructs the posterior GP kernel induced by the weight-space Laplace posterior.

    Args:
        model_fn (Callable): A function representing the model whose kernel is computed.
        mean (PyTree): The mean parameters of the model.
        posterior (Callable): A function returning the posterior state and covariance.
        prior_arguments (dict): Arguments required to initialize the posterior.
        **kwargs: Additional optional arguments:
            - dense (bool): Whether to return a dense GP kernel matrix.
            - output_layout (optional): Layout specification for the dense matrix.

    Returns:
        Callable: A function that computes the posterior GP kernel between two inputs.
    """
    # Create posterior state and covariance
    posterior_state = posterior(**prior_arguments)
    cov_mv = posterior_state.cov_mv(posterior_state.state)

    # Pushforward functions
    def pf_jvp(input: InputArray, vector: Params) -> PredArray:
        return jax.jvp(
            lambda p: model_fn(input=input, params=p),
            (mean,),
            (vector,),
        )[1]

    def pf_vjp(input: InputArray, vector: PredArray) -> Params:
        out, vjp_fun = jax.vjp(lambda p: model_fn(input=input, params=p), mean)
        return vjp_fun(vector.reshape(out.shape))

    def create_kernel_mv(x1: jax.Array, x2: jax.Array):
        def mv(vec: PredArray) -> PredArray:
            return pf_jvp(x1, cov_mv(pf_vjp(x2, vec)[0]))

        return mv

    if kwargs.get("dense", False):
        output_layout = kwargs.get("output_layout")
        if output_layout:
            return lambda x1, x2: util.mv.todense(
                create_kernel_mv(x1, x2), layout=output_layout
            )
        msg = (
            "Function should return a dense matrix, but no output layout is specified."
        )
        raise ValueError(msg)

    return create_kernel_mv
