"""Pushforward functions for weight space uncertainty.

This file contains the additional functions for pushing forward
weight space uncertainty onto output uncertainty.
"""

import math
from collections import OrderedDict

import jax
import jax.numpy as jnp

from laplax import util
from laplax.eval.utils import finalize_functions
from laplax.types import Callable, KeyType, PyTree
from laplax.util.ops import lmap, precompute_list

# -------------------------------------------------------------------------
# General utilities
# -------------------------------------------------------------------------


def pred_fn(**kwargs):
    return kwargs.get("pred")


def get_normal_weight_samples(
    key: KeyType,
    mean: PyTree,
    scale_mv: Callable,
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


def mc_pred_mean_fn(**kwargs):
    return util.tree.mean(kwargs.get("pred_ensemble"), axis=0)


def mc_pred_cov_fn(**kwargs):
    p_ens = kwargs.get("pred_ensemble")
    return util.tree.cov(p_ens.reshape(p_ens.shape[0], -1), rowvar=False)


def mc_pred_var_fn(**kwargs):
    """Compute variance of ensemble.

    Dependent compute:
    - (possible) pred_cov exists.
    - (fallback) pred_ensemble
    """
    cov = kwargs.get("pred_cov")
    return (
        jnp.diagonal(cov)
        if cov is not None
        else util.tree.var(kwargs.get("pred_ensemble"), axis=0)
    )


def mc_pred_std_fn(**kwargs):
    """Compute std of ensemble.

    Dependent compute:
    - (notimplementederror) pred_cov exists.
    - (possible) pred_var exists. (?)
    - (fallback) pred_ensemble
    """
    pred_var = kwargs.get("pred_var")
    return (
        jnp.sqrt(pred_var)
        if pred_var is not None
        else util.tree.std(kwargs.get("pred_ensemble"), axis=0)
    )


def mc_samples_fn(n_samples: int = 5, **kwargs):
    """Select samples from ensemble."""
    return util.tree.tree_slice(kwargs.get("pred_ensemble"), 0, n_samples)


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
    model_fn: Callable,
    mean: PyTree,
    posterior: Callable,
    prior_arguments: dict,
    n_weight_samples: int,
    pushforward_functions: dict = DEFAULT_MC_FUNCTIONS,
    **kwargs,
) -> Callable:
    # Create weight sample function
    posterior_state = posterior(**prior_arguments)
    scale_mv = posterior_state["scale_mv"](posterior_state["state"])

    get_weight_sample = set_get_weight_sample(
        key,
        mean,
        scale_mv,
        n_weight_samples,
    )

    # Create prob predictive function
    def prob_predictive(input):
        def compute_pred_ptw(idx):
            weight_sample = get_weight_sample(idx)
            return model_fn(params=weight_sample, input=input)

        pred = model_fn(params=mean, input=input)
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


def lin_pred_var_fn(**kwargs):
    # Get arguments
    cov = kwargs.get("pred_cov", kwargs.get("cov_mv"))  # Covariance
    pred = kwargs.get("pred")

    # Compute diagonal as variance
    var = util.mv.diagonal(
        cov, layout=math.prod(pred.shape)
    )  # This assumes output is not a tree.
    return var


def lin_pred_std_fn(**kwargs):
    var = kwargs.get("pred_var", lin_pred_var_fn(**kwargs))
    return util.tree.sqrt(var)


def lin_pred_cov_fn(**kwargs):
    # Get arguments
    cov_mv = kwargs.get("cov_mv")
    pred = kwargs.get("pred")

    # Compute diagonal as variance
    cov = util.mv.todense(cov_mv, layout=pred)
    return cov


def lin_n_samples_fn(n_samples: int = 5, **kwargs):
    # Get parameters
    scale_mv = kwargs.get("scale_mv")
    weight_samples = kwargs.get("weight_samples")

    # Get weight samples
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


def set_output_cov_mv(posterior_state, input, jvp, vjp):
    cov_mv = posterior_state["cov_mv"](posterior_state["state"])
    scale_mv = posterior_state["scale_mv"](posterior_state["state"])

    def output_cov_mv(vec):
        return jvp(input, cov_mv(vjp(input, vec)[0]))

    def output_cov_scale_mv(vec):
        return jvp(input, scale_mv(vec))

    return {"cov_mv": output_cov_mv, "scale_mv": output_cov_scale_mv}


def set_lin_pushforward(
    key: KeyType,
    model_fn: Callable,
    mean: PyTree,
    posterior: Callable,
    prior_arguments: dict,
    pushforward_functions: OrderedDict = DEFAULT_LIN_FINALIZE,
    **kwargs,
) -> Callable:
    # Create posterior state
    posterior_state = posterior(**prior_arguments)

    # Create pushforward functions
    def pf_jvp(input, vector):
        return jax.jvp(
            lambda p: model_fn(params=p, input=input),
            (mean,),
            (vector,),
        )[1]

    def pf_vjp(input, vector):
        out, vjp_fun = jax.vjp(lambda p: model_fn(params=p, input=input), mean)
        return vjp_fun(vector.reshape(out.shape))

    # Create scale mv
    if "samples" in pushforward_functions:
        n_samples = kwargs.pop("n_samples")
        get_weight_samples = set_get_weight_sample(
            key,
            mean,
            posterior_state["scale_mv"](posterior_state["state"]),
            n_samples,
            **kwargs,
        )
    else:
        get_weight_samples = None
        n_samples = 0

    def prob_predictive(input: jax.Array):
        # Mean prediction
        pred = model_fn(params=mean, input=input)

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
    model_fn: Callable,
    mean: PyTree,
    posterior: Callable,
    prior_arguments: dict,
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
    cov_mv = posterior_state["cov_mv"](posterior_state["state"])

    # Pushforward functions
    def pf_jvp(input, vector):
        return jax.jvp(
            lambda p: model_fn(params=p, input=input),
            (mean,),
            (vector,),
        )[1]

    def pf_vjp(input, vector):
        out, vjp_fun = jax.vjp(lambda p: model_fn(params=p, input=input), mean)
        return vjp_fun(vector.reshape(out.shape))

    def create_kernel_mv(x1: jax.Array, x2: jax.Array):
        def mv(vec: jax.Array):
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
