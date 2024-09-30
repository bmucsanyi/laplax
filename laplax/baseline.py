"""Baseline for Laplace approximation.

This files contain implementations for input and weight space perturbations.
The design principle is given by the following structure:

input perturbations:
- model_fn : x -> model(x, p)
- get_mean : x -> x
- get_cov_scale : tau -> tau * Id
- get_prob_predictive : x -> model(x + eps, p) for eps ~ N(0, tau * Id)

"""

from collections.abc import Callable
from copy import deepcopy
from functools import partial
from math import prod

import jax
import jax.numpy as jnp

from laplax.config import lmap

# --------------------------------------------------------------------
# BASELINE: Input perturbations
# --------------------------------------------------------------------


def set_prob_predictive_with_input_perturbations(
    model, prior_scale, input_shape=None, **kwargs
):
    # Get relevant hyperparameters
    rng = kwargs.get("key")
    n_weight_samples = kwargs.get("n_weight_samples")
    mode = kwargs.get("mode", "metric")
    pre_sample = kwargs.get("pre_sample", False)
    if pre_sample and input_shape is None:
        msg = "Input shape must be provided for pre-sampling."
        raise ValueError(msg)

    # Define random sampling for mc sampling
    rng_weights = jax.random.split(rng, n_weight_samples)
    if pre_sample:
        wight_noise = prior_scale * jax.vmap(
            lambda key: jax.random.normal(key, (*input_shape,))
        )(rng_weights)

    def get_prob_predictive(input: jax.Array):
        # Define function to generate ptw predictions
        def single_sample_predictions(idx: int):
            if pre_sample:
                return model(input + wight_noise[idx])
            return model(
                input
                + prior_scale * jax.random.normal(rng_weights[idx], (*input.shape,))
            )

        # Ensemble predictions
        pred_ensemble = lmap(single_sample_predictions, jnp.arange(n_weight_samples))

        # General prediction
        pred = model(input)

        return {
            "pred": pred,
            "pred_mean": jax.numpy.mean(pred_ensemble, axis=0),
            "pred_std": jax.numpy.std(pred_ensemble, axis=0),
            "pred_ensemble": pred_ensemble if mode == "ensemble" else None,
        }

    return get_prob_predictive


def initialize_input_perturbations(model, input_shape, **kwargs):
    # set model_fn : x -> model(x, p)
    def model_fn(x):
        return model(x)

    # set get_mean : x -> x
    def get_mean(x):
        return x

    # set get_cov_scale : tau -> tau * Id
    def get_cov_scale(tau):
        return tau

    # set set_prob_predictive : x -> model(x + eps, p) for eps ~ N(0, tau * Id)
    set_prob_predictive = partial(
        set_prob_predictive_with_input_perturbations,
        model=model,
        input_shape=input_shape,
        **kwargs,
    )

    return {
        "model_fn": model_fn,
        "get_mean": get_mean,
        "get_cov_scale": get_cov_scale,
        "set_prob_predictive": set_prob_predictive,
    }


# --------------------------------------------------------------------
# BASELINE: Weight space perturbations
# --------------------------------------------------------------------


def set_prob_predictive_with_weight_perturbations(
    model_fn: Callable,
    prior_scale: float,
    relevant_params: list[jax.Array],
    set_params: dict,
    **kwargs,
):
    """Return prob-predictions for weight perturbations."""
    # Get relevant hyperparameters
    rng = kwargs.get("rng", jax.random.PRNGKey(0))
    n_weight_samples = kwargs.get("n_weight_samples", 100)
    mode = kwargs.get("mode", "metric")
    pre_sample = kwargs.get("pre_sample", False)

    # Get model subset behavior
    params_true = deepcopy(relevant_params)
    param_shapes = [p.shape for p in relevant_params]
    n_params = 0
    split_indices = [n_params := n_params + prod(shape) for shape in param_shapes][:-1]

    def flat_to_list_split(flat_params):
        return jnp.split(flat_params, split_indices)

    def perturb_params(key):
        return deepcopy(
            set_params(
                [
                    p + eps.reshape(p.shape)
                    for p, eps in zip(
                        params_true,
                        flat_to_list_split(
                            prior_scale * jax.random.normal(key, (n_params,))
                        ),
                        strict=True,
                    )
                ],
            )
        )

    # Define random sampling
    rng_weights = jax.random.split(rng, n_weight_samples)
    if pre_sample:
        params_perturbed = jax.vmap(perturb_params)(rng_weights)

    # Set prob predictive
    def get_prob_predictions(input: jax.Array) -> dict[jax.Array]:
        # Define function to generate ptw predictions
        def single_sample_predictions(idx: int):
            if pre_sample:
                return model_fn(input=input, params=params_perturbed[idx])
            return model_fn(
                input=input,
                params=perturb_params(rng_weights[idx]),
            )

        # Ensemble predictions
        pred_ensemble = lmap(single_sample_predictions, jnp.arange(n_weight_samples))

        # General prediction
        pred = model_fn(input=input, params=set_params(params_true))

        return {
            "pred": pred,
            "pred_mean": jax.numpy.mean(pred_ensemble, axis=0),
            "pred_std": jax.numpy.std(pred_ensemble, axis=0),
            "pred_ensemble": pred_ensemble if mode == "ensemble" else None,
        }

    return get_prob_predictions
