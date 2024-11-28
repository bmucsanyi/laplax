# noqa: D100
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import jax

from laplax.util.ops import lmap
from laplax.util.util import identity


def finalize_functions(functions: OrderedDict, results: dict, **kwargs):
    """Finalize functions.

    Scans over ordered dictionary of functions (metrics, pushforwards, ...) and fills
    in the results dictionary. Relevant function values are passed as kwargs.
    """
    for name, func in functions.items():
        results[name] = func(**results, **kwargs)
    return results


# def get_predictions_for_data_point_fn():
#     def get_predictions_for_data_point(
#         x: jax.Array,  # noqa: ARG001
#     ) -> tuple[jax.Array, jax.Array]:
#         return

#     return get_predictions_for_data_point


def evaluate_metrics_on_dataset(  # noqa: D417
    pred_fn: Callable,
    data: tuple[jax.Array],
    *,
    metrics: OrderedDict[Callable],
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
        pred = {**pred_fn(dp["input"]), "target": dp["target"]}
        return finalize_functions(functions=metrics, results={}, **pred)

    evaluated_metrics = lmap(evaluate_data_point, data)
    return {metric: apply(evaluated_metrics[metric]) for metric in evaluated_metrics}
