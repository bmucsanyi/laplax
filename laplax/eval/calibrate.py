# noqa: D100
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from laplax.eval.metrics import estimate_q
from laplax.util.ops import laplax_dtype, lmap


# Calibrate prior
def calibration_metric(**predictions):
    return jnp.abs(estimate_q(**predictions) - 1)


def evaluate_for_given_prior_arguments(
    *,
    data: tuple[jax.Array, jax.Array],
    set_prob_predictive=Callable,
    metric=calibration_metric,
    **kwargs,
):
    prob_predictive = set_prob_predictive(**kwargs)

    def evaluate_data(dp: tuple[jax.Array]) -> dict:
        return {**prob_predictive(dp["input"]), "target": dp["target"]}

    res = metric(
        **lmap(
            evaluate_data, data, batch_size=kwargs.get("lmap_eval_prior_prec", "data")
        )
    )
    return res


def grid_search(
    prior_prec_interval: jax.Array,
    objective: Callable[[float, tuple[jax.Array, jax.Array]], float],
    data: tuple[jax.Array, jax.Array],
    patience: int = 5,
    max_iterations: int | None = None,
) -> float:
    results, prior_precs = [], []
    increasing_count = 0
    previous_result = None

    for iteration, prior_prec in enumerate(prior_prec_interval):
        start_time = time.perf_counter()
        try:
            result = objective(prior_arguments={"prior_prec": prior_prec}, data=data)
        except ValueError as error:
            print(f"Caught an exception in validate: {error}")  # noqa: T201
            result = float("inf")

        if jnp.isnan(result):
            print("Caught nan, setting result to inf.")  # noqa: T201
            result = float("inf")

        # Logging for performance and tracking
        print(  # noqa: T201
            f"Took {time.perf_counter() - start_time:.4f} seconds, "
            f"prior prec: {prior_prec:.4f}, result: {result:.6f}"
        )

        results.append(result)
        prior_precs.append(prior_prec)

        # If we have a previous result, check if the result has increased
        if previous_result is not None:
            if result > previous_result:
                increasing_count += 1
                print(f"Result increased, increasing_count = {increasing_count}")  # noqa: T201
            else:
                increasing_count = 0

            # Stop if the results have increased for `patience` consecutive iterations
            if increasing_count >= patience:
                break

        previous_result = result

        # Check if maximum iterations reached
        if max_iterations is not None and iteration >= max_iterations:
            print(f"Stopping due to reaching max iterations: {max_iterations}")  # noqa: T201
            break

    best_prior_prec = prior_precs[np.nanargmin(results)]
    print(f"Chosen prior prec: {best_prior_prec:.4f}")  # noqa: T201

    return best_prior_prec


def optimize_prior_prec(
    objective: Callable[[float, tuple[jax.Array, jax.Array]], float],
    data: tuple[jax.Array, jax.Array],
    log_prior_prec_min: float = -5.0,
    log_prior_prec_max: float = 6.0,
    grid_size: int = 300,
) -> float:
    prior_prec_interval = jnp.logspace(
        start=log_prior_prec_min,
        stop=log_prior_prec_max,
        num=grid_size,
        dtype=laplax_dtype(),
    )
    prior_prec = grid_search(
        prior_prec_interval,
        objective,
        data,
    )

    return prior_prec
