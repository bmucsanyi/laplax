# noqa: D100
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from laplax.config import laplax_dtype, lmap
from laplax.eval.metrics import estimate_q


# Calibrate prior
def calibration_metric(**predictions):
    return jnp.abs(estimate_q(**predictions) - 1)


def evaluate_for_given_prior_prec(
    prior_prec: float,
    data,
    set_prob_predictive=Callable,
    metric=calibration_metric,
):
    prob_predictive = set_prob_predictive(prior_scale=prior_prec)

    def evaluate_data(dp):
        input, target = dp
        return {**prob_predictive(input), "target": target}

    res = metric(**lmap(evaluate_data, (data[0], data[1])))
    return res


# ------------------------------------------------------------------------------
# Alternative optimization
# ------------------------------------------------------------------------------

# Gradient-based optimization


# ------------------------------------------------------------------------------
# Grid search
# ------------------------------------------------------------------------------


def grid_search(
    prior_prec_interval: jax.Array,
    objective: Callable[[float, tuple[jax.Array, jax.Array]], float],
    data: tuple[jax.Array, jax.Array],
) -> float:
    results, prior_precs = [], []
    for prior_prec in prior_prec_interval:
        start_time = time.perf_counter()
        try:
            result = objective(prior_prec, data)
        except ValueError as error:
            print(f"Caught an exception in validate: {error}")  # noqa: T201
            result = float("inf")
        if jnp.isnan(result):  # TODO(2bys): Check if we want this.
            print("Caught nan, setting result to inf.")  # noqa: T201
            result = float("inf")
        print(  # noqa: T201
            f"Took {time.perf_counter() - start_time:.4f} seconds, "
            f"prior prec: {prior_prec:.4f}, result: {result:.6f}"
        )
        results.append(result)
        prior_precs.append(prior_prec)

    best_prior_prec = prior_precs[np.nanargmin(results)]

    print(f"Chosen prior prec: {best_prior_prec:.4f}")  # noqa: T201

    return best_prior_prec


def optimize_prior_prec(
    objective: Callable[[float, tuple[jax.Array, jax.Array]], float],
    data: tuple[jax.Array, jax.Array],
    log_prior_prec_min: float = -5.0,
    log_prior_prec_max: float = 4.0,
    grid_size: int = 100,
) -> float:
    prior_prec_interval = jnp.logspace(
        start=log_prior_prec_min,
        stop=log_prior_prec_max,
        num=grid_size,
        dtype=laplax_dtype(),
    )
    prior_prec = grid_search(prior_prec_interval, objective, data)

    return prior_prec
