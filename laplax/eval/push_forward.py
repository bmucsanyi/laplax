"""Push-forward functions for weight space uncertainty.

This file contains the additional functions for pushing forward
weight space uncertainty onto output uncertainty.
"""

import functools

import jax
import jax.numpy as jnp

from laplax.types import Callable, KeyType, PyTree
from laplax.util.ops import lmap, precompute_list
from laplax.util.tree import add, cov, mean, randn_like, std, var

# -------------------------------------------------------------------------
# General utilities
# -------------------------------------------------------------------------


def finalize_functions(func_dict, bool_dict):
    selected_functions = [v for k, v in func_dict.items() if bool_dict[k]]

    def fn(**kwargs):
        results = {}
        for func in selected_functions:
            results.update(func(**kwargs))
        return results

    return fn


# -------------------------------------------------------------------------
# Monte Carlo push-forward
# -------------------------------------------------------------------------


def get_normal_weight_samples(
    key: KeyType,
    mean: PyTree,
    scale_mv: Callable,
) -> PyTree:
    # Draw white noise
    noise = randn_like(key, mean)

    # Apply scale mv
    return add(mean, scale_mv(noise))


def mc_pred_fn(**kwargs):
    pred = kwargs.get("pred")
    return {"pred": pred}


def mc_pred_mean_fn(**kwargs):
    pred_ensemble = kwargs.get("pred_ensemble")
    return {"pred_mean": mean(pred_ensemble, axis=0)}


def mc_pred_std_fn(**kwargs):
    pred_ensemble = kwargs.get("pred_ensemble")
    return {"pred_std": std(pred_ensemble, axis=0)}


def mc_pred_cov_fn(**kwargs):
    pred_ensemble = kwargs.get("pred_ensemble")
    return {"pred_cov": cov(pred_ensemble, rowvar=False)}


def mc_pred_var_fn(**kwargs):
    pred_ensemble = kwargs.get("pred_ensemble")
    return {"pred_var": var(pred_ensemble, axis=0)}


mc_finalize_pushforward = {
    "compute_pred": mc_pred_fn,
    "compute_pred_mean": mc_pred_mean_fn,
    "compute_pred_std": mc_pred_std_fn,
    "compute_pred_cov": mc_pred_cov_fn,
    "compute_pred_var": mc_pred_var_fn,
}

mc_finalize_pushforward_default_boolean = {
    "compute_pred": True,
    "compute_pred_mean": True,
    "compute_pred_std": True,
    "compute_pred_cov": False,
    "compute_pred_var": False,
    "compute_n_samples": None,
}


def set_mc_pushforward(
    key: KeyType,
    model_fn: Callable,
    mean: PyTree,
    posterior: Callable,
    prior_prec: float,
    n_weight_samples: int,
    pushforward_functions: dict = mc_finalize_pushforward,
    pushforward_default_boolean: dict = mc_finalize_pushforward_default_boolean,
    **kwargs,
) -> Callable:
    # Merge default booleans with kwargs, giving priority to kwargs
    kwargs = {**pushforward_default_boolean, **kwargs}
    finalize = finalize_functions(pushforward_functions, kwargs)

    # Create sample function
    keys = jax.random.split(key, n_weight_samples)
    scale_mv = posterior(prior_prec, return_scale=True)

    def get_weight_sample(idx):
        return get_normal_weight_samples(keys[idx], mean, scale_mv)

    get_weight_sample = precompute_list(get_weight_sample, jnp.arange(n_weight_samples))

    # Create prob predictive function
    def prob_predictive(input):
        def compute_pred_ptw(idx):
            weight_sample = get_weight_sample(idx)
            return model_fn(params=weight_sample, input=input)

        pred = model_fn(params=mean, input=input)
        pred_ensemble = lmap(compute_pred_ptw, jnp.arange(n_weight_samples))

        return finalize(pred=pred, pred_ensemble=pred_ensemble)

    return prob_predictive
