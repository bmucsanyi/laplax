import logging
import time
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np

from laplax.eval.metrics import estimate_q
from laplax.types import Array, Data, Float, PriorArguments
from laplax.util.ops import laplax_dtype, lmap


# Calibrate prior
def calibration_metric(**predictions) -> Float:
    return jnp.abs(estimate_q(**predictions) - 1)


def evaluate_for_given_prior_arguments(
    *,
    data: Data,
    set_prob_predictive: Callable,
    metric: Callable = calibration_metric,
    **kwargs,
):
    prob_predictive = set_prob_predictive(**kwargs)

    def evaluate_data(dp: Data) -> dict[str, Array]:
        return {**prob_predictive(dp["input"]), "target": dp["target"]}

    res = metric(
        **lmap(
            evaluate_data, data, batch_size=kwargs.get("lmap_eval_prior_prec", "data")
        )
    )
    return res


def grid_search(
    prior_prec_interval: Array,
    objective: Callable[[PriorArguments], float],
    patience: int = 5,
    max_iterations: int | None = None,
) -> Float:
    results, prior_precs = [], []
    increasing_count = 0
    previous_result = None

    for iteration, prior_prec in enumerate(prior_prec_interval):
        start_time = time.perf_counter()
        try:
            result = objective({"prior_prec": prior_prec})
        except ValueError as error:
            logging.warning("Caught an exception in validate: %s", error)
            result = float("inf")

        if jnp.isnan(result):
            logging.info("Caught nan, setting result to inf.")
            result = float("inf")

        # Logging for performance and tracking
        logging.info(
            "Took %.4f seconds, prior prec: %.4f, result: %.6f",
            time.perf_counter() - start_time,
            prior_prec,
            result,
        )

        results.append(result)
        prior_precs.append(prior_prec)

        # If we have a previous result, check if the result has increased
        if previous_result is not None:
            if result > previous_result:
                increasing_count += 1
                logging.info(
                    "Result increased, increasing_count = %d", increasing_count
                )
            else:
                increasing_count = 0

            # Stop if the results have increased for `patience` consecutive iterations
            if increasing_count >= patience:
                break

        previous_result = result

        # Check if maximum iterations reached
        if max_iterations is not None and iteration >= max_iterations:
            logging.info("Stopping due to reaching max iterations: %d", max_iterations)
            break

    best_prior_prec = prior_precs[np.nanargmin(results)]
    logging.info("Chosen prior prec: %.4f", best_prior_prec)

    return best_prior_prec


def optimize_prior_prec(
    objective: Callable[[PriorArguments], float],
    log_prior_prec_min: float = -5.0,
    log_prior_prec_max: float = 6.0,
    grid_size: int = 300,
) -> Float:
    prior_prec_interval = jnp.logspace(
        start=log_prior_prec_min,
        stop=log_prior_prec_max,
        num=grid_size,
        dtype=laplax_dtype(),
    )
    prior_prec = grid_search(
        prior_prec_interval,
        objective,
    )

    return prior_prec
