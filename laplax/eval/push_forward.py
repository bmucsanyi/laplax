"""Push-forward functions for weight space uncertainty.

This file contains the additional functions for pushing forward
weight space uncertainty onto output uncertainty.
"""

import jax
from jax import random
from jaxtyping import PyTree

# --------------------------------------------------------------------------------
# Monte Carlo predictions
# --------------------------------------------------------------------------------


def create_mc_predictions_for_data_point_fn(model_fn, mean, cov_scale, param_builder):
    rng_key, _ = jax.random.PRNGKey(42)
    samples = random.multivariate_normal(rng_key, mean, cov_scale, (1000,))

    def get_predictions_for_data_point(data_point: jax.Array):
        def pred_fn(p: PyTree) -> jax.Array:
            return model_fn(param_builder(p), data_point)

        la_pred = jax.vmap(pred_fn)(samples)
        model_pred = pred_fn(mean)
        return {
            "pred": model_pred,
            "pred_mean": la_pred.mean(axis=0),
            "pred_std": la_pred.std(axis=0),
        }

    return get_predictions_for_data_point


# --------------------------------------------------------------------------------
# Linearization
# --------------------------------------------------------------------------------
