# noqa: D100
from collections.abc import Callable
from typing import Any

import jax


def get_predictions_for_data_point_fn():  # noqa: ANN201, D103
    def get_predictions_for_data_point(
        x: jax.Array,  # noqa: ARG001
    ) -> tuple[jax.Array, jax.Array]:
        return

    return get_predictions_for_data_point


def get_evaluation_fn():  # noqa: ANN201, D103
    def evaluate_dataset(data: tuple[jax.Array, jax.Array]) -> float:  # noqa: ARG001
        return

    return evaluate_dataset


def get_calibration_evaluation_fn():  # noqa: ANN201, D103
    def evaluate_dataset(data: tuple[jax.Array, jax.Array]) -> float:  # noqa: ARG001
        return

    return evaluate_dataset


def log_metrics():  # noqa: ANN201, D103
    return ...


def _aslist(obj: Any) -> list[Any]:  # noqa: ANN401
    return obj if isinstance(obj, list) else [obj]


def evaluate_metrics_on_dataset(
    pred_fn: Callable, data: tuple[jax.Array], *, metrics: list[Callable] | Callable
) -> dict:
    """Evaluate metrics on a dataset.

    Args:
        pred_fn: A callable that takes an input and returns predictions.
        data: A tuple of input data and target data.
        metrics: A list of callable metrics or a single callable metric.

    Returns:
        A dictionary containing the evaluated metrics.

    """

    def evaluate_data_point(x: tuple[jax.Array]) -> dict:
        return {**pred_fn(x[0]), "target": x[1]}

    return {
        metric.__name__: jax.vmap(lambda d: metric(**evaluate_data_point(d)))(data)  # noqa: B023
        for metric in _aslist(metrics)
    }
