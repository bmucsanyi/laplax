# noqa: D100
import argparse
import time
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from laplax.eval.utils import compute_calibration_metric, get_calibration_evaluation_fn
from laplax.types import CalibrationMetric, Params


def get_calibration_objective_fn(  # noqa: D103
    args: argparse.Namespace,
    model_fn: Callable,
    params: Params,
    get_cov: Callable[[float], jax.Array],
    metric: CalibrationMetric,
) -> Callable[[float, tuple[jax.Array, jax.Array]], float]:
    def evaluate_for_given_prior_prec(
        prior_prec: float, data: tuple[jax.Array, jax.Array]
    ) -> float:
        cov = get_cov(prior_prec)

        evaluate_dataset = get_calibration_evaluation_fn(
            args=args,
            model_fn=model_fn,
            params=params,
            cov=cov,
            compute_metrics=partial(compute_calibration_metric, metric=metric),
        )

        return evaluate_dataset(data)

    return evaluate_for_given_prior_prec


def grid_search(  # noqa: D103
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
        if jnp.isnan(result):  # TODO(2bys): Check if we want this.  # noqa: TD003
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


def optimize_prior_prec(  # noqa: D103
    objective: Callable[[float, tuple[jax.Array, jax.Array]], float],
    data: tuple[jax.Array, jax.Array],
    log_prior_prec_min: float = -3.0,
    log_prior_prec_max: float = 6.0,
    grid_size: int = 100,
) -> float:
    prior_prec_interval = jnp.logspace(
        start=log_prior_prec_min, stop=log_prior_prec_max, num=grid_size
    )
    prior_prec = grid_search(prior_prec_interval, objective, data)

    return prior_prec
