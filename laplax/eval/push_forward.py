"""Push-forward functions for weight space uncertainty.

This file contains the additional functions for pushing forward
weight space uncertainty onto output uncertainty.
"""

import math
from collections import OrderedDict

import jax
import jax.numpy as jnp

from laplax import util
from laplax.types import Callable, KeyType, PyTree
from laplax.util.ops import lmap, precompute_list

# -------------------------------------------------------------------------
# General utilities
# -------------------------------------------------------------------------


def finalize_functions(functions: OrderedDict, results: dict, **kwargs):
    for name, func in functions.items():
        results[name] = func(**results, **kwargs)
    return results


def pred_fn(**kwargs):
    return kwargs.get("pred")


def get_normal_weight_samples(
    key: KeyType,
    mean: PyTree,
    scale_mv: Callable,
) -> PyTree:
    return util.tree.add(mean, scale_mv(util.tree.randn_like(key, mean)))


def set_get_weight_sample(key, mean, scale_mv, n_weight_samples):
    keys = jax.random.split(key, n_weight_samples)

    def get_weight_sample(idx):
        return get_normal_weight_samples(keys[idx], mean, scale_mv)

    return precompute_list(get_weight_sample, jnp.arange(n_weight_samples))


# -------------------------------------------------------------------------
# Monte Carlo push-forward
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
        else util.tree.var(kwargs.get("pred_ensemble"))
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


def set_mc_pushforward(  # noqa: PLR0913, PLR0917
    key: KeyType,
    model_fn: Callable,
    mean: PyTree,
    posterior: Callable,
    prior_prec: float,
    n_weight_samples: int,
    mc_pushforward_functions: dict = DEFAULT_MC_FUNCTIONS,
) -> Callable:
    # Create weight sample function
    scale_mv = posterior(prior_prec=prior_prec, return_scale=True)["scale_mv"]
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
        pred_ensemble = lmap(compute_pred_ptw, jnp.arange(n_weight_samples))

        return finalize_functions(
            functions=mc_pushforward_functions,
            results={"pred": pred},
            pred_ensemble=pred_ensemble,
        )

    return prob_predictive


# -------------------------------------------------------------------------
# Linearized push-forward
# -------------------------------------------------------------------------


def lin_pred_var_fn(**kwargs):
    # Get arguments
    _cov = kwargs.get("pred_cov", kwargs.get("cov_mv"))  # Covariance
    pred = kwargs.get("pred")

    # Compute diagonal as variance
    _var = util.mv.diagonal(
        _cov, size=math.prod(pred.shape)
    )  # This assumes output is not a tree.
    return _var


def lin_pred_std_fn(**kwargs):
    _var = kwargs.get("pred_var", lin_pred_var_fn(**kwargs))
    return util.tree.std(_var)


def lin_pred_cov_fn(**kwargs):
    # Get arguments
    cov_mv = kwargs.get("cov_mv")
    pred = kwargs.get("pred")

    # Compute diagonal as variance
    cov = util.mv.todense(cov_mv, like=pred)
    return cov


def lin_n_samples_fn(n_samples: int = 5, **kwargs):
    # Get parameters
    scale_mv = kwargs.get("scale_mv")
    weight_samples = kwargs.get("weight_samples")

    # Get weight samples
    return lmap(lambda i: scale_mv(weight_samples(i)), jnp.arange(n_samples))


DEFAULT_LIN_FINALIZE = OrderedDict([
    ("pred", pred_fn),
    ("pred_mean", pred_fn),
    ("pred_var", lin_pred_var_fn),
    ("pred_std", lin_pred_std_fn),
    ("pred_cov", lin_pred_cov_fn),
    ("samples", lin_n_samples_fn),
])


def set_output_cov_mv(mv, input, jvp, vjp):
    def cov_mv(vec):
        return jvp(input, mv.get("cov_mv")(vjp(input, vec)[0]))

    def cov_scale_mv(vec):
        return jvp(input, mv.get("scale_mv")(vec))

    return {"cov_mv": cov_mv, "scale_mv": cov_scale_mv if mv.get("scale_mv") else None}


def set_lin_pushforward(  # noqa: PLR0913, PLR0917
    key: KeyType,
    model_fn: Callable,
    mean: PyTree,
    posterior: Callable,
    prior_prec: float,
    linearized_pushforward_functions: OrderedDict = DEFAULT_LIN_FINALIZE,
    **kwargs,
) -> Callable:
    # Create mv function
    mv = posterior(
        prior_prec=prior_prec,
        return_scale=("samples" in linearized_pushforward_functions),
    )

    # Create push-forward functions
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
    if "samples" in linearized_pushforward_functions:
        n_samples = kwargs.get("n_samples")
        get_weight_samples = set_get_weight_sample(
            key, mean, mv.get("scale_mv"), n_samples
        )
    else:
        get_weight_samples = None
        n_samples = 0

    def prob_predictive(input: jax.Array):
        # Mean prediction
        pred = model_fn(params=mean, input=input)

        # Compute prediction
        return finalize_functions(
            functions=linearized_pushforward_functions,
            results={"pred": pred},
            **set_output_cov_mv(mv, input=input, jvp=pf_jvp, vjp=pf_vjp),
            input=input,
            weight_samples=get_weight_samples,
            n_samples=n_samples,
        )

    return prob_predictive
