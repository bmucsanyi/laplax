# noqa: D100
from collections.abc import Callable
from typing import Any

import jax


def get_predictions_for_data_point_fn():
    def get_predictions_for_data_point(
        x: jax.Array,  # noqa: ARG001
    ) -> tuple[jax.Array, jax.Array]:
        return

    return get_predictions_for_data_point


def _aslist(obj: Any) -> list[Any]:
    return obj if isinstance(obj, list) else [obj]


def identity(x: Any) -> Any:
    return x


def evaluate_metrics_on_dataset(  # noqa: D417
    pred_fn: Callable,
    data: tuple[jax.Array],
    *,
    metrics: list[Callable] | Callable,
    apply: Callable = identity,
) -> dict:
    """Evaluate metrics on a dataset.

    Args:
        pred_fn: A callable that takes an input and returns predictions.
        data: A tuple of input data and target data.
        metrics: A list of callable metrics or a single callable metric.

    Returns:
        A dictionary containing the evaluated metrics.

    """

    def evaluate_data_point(dp: tuple[jax.Array]) -> dict:
        pred = {**pred_fn(dp[0]), "target": dp[1]}
        return {metric.__name__: metric(**pred) for metric in _aslist(metrics)}

    evaluated_metrics = jax.vmap(evaluate_data_point)(data)
    return {metric: apply(evaluated_metrics[metric]) for metric in evaluated_metrics}
