# noqa: D100
from collections import OrderedDict

import jax

from laplax.types import Callable
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


def evaluate_on_dataset(pred_fn: Callable, data: tuple[jax.Array], **kwargs) -> dict:
    """Evaluate prob_predictive on dataset.

    Args:
        pred_fn: A callable that takes an input and returns predictions.
        data: A tuple of input and target data.
        metircs: A list of callable metrics or a single callable metric.
        **kwargs:
            - lmap_eval: lmap batchsize
    Returns:
        A dictionary containing predictions.
    """

    def evaluate_data_point(dp: tuple[jax.Array]) -> dict:
        return {**pred_fn(dp["input"]), "target": dp["target"]}

    return lmap(evaluate_data_point, data, batch_size=kwargs.get("lmap_eval", "data"))


def evaluate_metrics_on_dataset(  # noqa: D417
    pred_fn: Callable,
    data: tuple[jax.Array],
    *,
    metrics: OrderedDict[Callable],
    apply: Callable = identity,
    **kwargs,
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

    evaluated_metrics = lmap(
        evaluate_data_point, data, batch_size=kwargs.get("lmap_eval_metrics", "data")
    )
    return {metric: apply(evaluated_metrics[metric]) for metric in evaluated_metrics}
