"""Push-forward functions for weight space uncertainty.

This file contains the additional functions for pushing forward
weight space uncertainty onto output uncertainty.
"""

import jax
import jax.numpy as jnp

from laplax.types import Callable, KeyType, PyTree
from laplax.util.ops import lmap, precompute_list
from laplax.util.tree import add, randn_like, std


def get_normal_weight_samples(
    key: KeyType,
    mean: PyTree,
    scale_mv: Callable,
) -> PyTree:
    # Draw white noise
    noise = randn_like(key, mean)

    # Apply scale mv
    return add(mean, scale_mv(noise))


def set_mc_pushforward(
    key: KeyType,
    model_fn: Callable,
    mean: PyTree,
    scale_mv: Callable,
    n_weight_samples: int,
) -> Callable:
    # Create sample function
    keys = jax.random.split(key, n_weight_samples)

    def get_weight_sample(idx):
        return get_normal_weight_samples(keys[idx], mean, scale_mv)

    get_weight_sample = precompute_list(get_weight_sample, jnp.arange(n_weight_samples))

    def prob_predictive(input):
        def func_ptw(idx):
            weight_sample = get_weight_sample(idx)
            return model_fn(params=weight_sample, input=input)

        pred = model_fn(params=mean, input=input)
        pred_ensemble = lmap(func_ptw, jnp.arange(n_weight_samples))

        return {
            "pred": pred,
            "pred_mean": pred_ensemble,
            "pred_std": std(pred_ensemble, axis=0),
        }

    return prob_predictive
