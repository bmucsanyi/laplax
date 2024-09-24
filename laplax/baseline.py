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

import jax
import jax.numpy as jnp

from laplax.config import lmap

# --------------------------------------------------------------------
# BASELINE: Input perturbations
# --------------------------------------------------------------------


def set_prob_predictive_with_input_perturbations(
    model, cov_scale, input_shape, **kwargs
):
    rng = kwargs.get("rng", jax.random.PRNGKey(0))
    n_weight_samples = kwargs.get("n_weight_samples", 100)
    mode = kwargs.get("mode", "metric")
    mc_samples = cov_scale * jax.random.normal(rng, (n_weight_samples, *input_shape))

    def get_prob_predictive(x):
        pred = model(x)
        pred_ensemble = lmap(model, x[None] + mc_samples)
        return {
            "pred": pred,
            "pred_mean": jax.numpy.mean(pred_ensemble, axis=0),
            "pred_std": jax.numpy.std(pred_ensemble, axis=0),
            "pred_ensemble": pred_ensemble if mode == "ensemble" else None,
        }

    return get_prob_predictive


def initialize_input_perturbations(model, input_shape):
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


def set_prob_predictive_with_weight_perturbations(  # noqa: PLR0913, PLR0917
    model_fn: Callable,
    cov_scale: float,
    params: dict,
    param_shapes: list[tuple],
    param_access: Callable,
    param_set: Callable,
    **kwargs,
):
    """Return prob-predictions for weight perturbations."""
    rng = kwargs.get("rng", jax.random.PRNGKey(0))
    n_weight_samples = kwargs.get("n_weight_samples", 100)
    mode = kwargs.get("mode", "metric")
    keys = jax.random.split(rng, n_weight_samples)
    mc_samples = [
        cov_scale * jax.random.normal(key, (n_weight_samples, *shape))
        for key, shape in zip(keys, param_shapes, strict=False)
    ]
    params_true = deepcopy(param_access(params))
    params_change = deepcopy(params)
    params_perturbed = []
    params_perturbed = [
        deepcopy(
            param_set(
                params_change,
                [p + eps for p, eps in zip(params_true, perturbs, strict=False)],
            )
        )
        for perturbs in zip(*mc_samples, strict=False)
    ]

    def get_prob_predictions(input):
        pred = model_fn(input=input, params=param_set(params_change, params_true))
        pred_ensemble = jnp.asarray([
            model_fn(input=input, params=p) for p in params_perturbed
        ])

        return {
            "pred": pred,
            "pred_mean": jax.numpy.mean(pred_ensemble, axis=0),
            "pred_std": jax.numpy.std(pred_ensemble, axis=0),
            "pred_ensemble": pred_ensemble if mode == "ensemble" else None,
        }

    return get_prob_predictions
