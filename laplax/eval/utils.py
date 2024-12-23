from laplax.types import Any, Array, Callable, Data, InputArray
from laplax.util.ops import lmap
from laplax.util.utils import identity


def finalize_function_wrapper(
    fn: Callable,
) -> Callable:
    def wrapper(
        results: dict[str, Array], aux: dict[str, Any] | None, name: str, **kwargs
    ):
        results[name] = fn(**kwargs)
        return results, aux

    return wrapper


def finalize_functions(
    functions: dict[str, Callable],
    results: dict,  # Typing must allow empty dict for initializations.
    aux: dict[str, Any] | None = None,
    **kwargs,
):
    """Finalize functions.

    Scans over dictionary of functions (metrics, pushforwards, ...) and fills
    in the results dictionary. Relevant function values are passed as kwargs.
    """
    # if aux is None:
    #     aux = {"none": None}
    for name, func in functions.items():
        results, aux = func(results=results, aux=aux, name=name, **kwargs)
    return results


def evaluate_on_dataset(
    pred_fn: Callable[[InputArray], dict[str, Array]], data: Data, **kwargs
) -> dict:
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

    def evaluate_data_point(dp: Data) -> dict[str, Array]:
        return {**pred_fn(dp["input"]), "target": dp["target"]}

    return lmap(evaluate_data_point, data, batch_size=kwargs.get("lmap_eval", "data"))


def evaluate_metrics_on_dataset(
    pred_fn: Callable[[InputArray], dict[str, Array]],
    data: Data,
    *,
    metrics: dict[str, Callable],
    apply: Callable = identity,
    **kwargs,
) -> dict:
    """Evaluate metrics on a dataset.

    Args:
        pred_fn: A callable that takes an input and returns predictions.
        data: A tuple of input data and target data.
        metrics: A list of callable metrics or a single callable metric.
        apply: Callable to apply to the evaluated metrics.
        kwargs: Additional keyword arguments.

    Returns:
        A dictionary containing the evaluated metrics.
    """
    # Wrap metrics
    metrics = {name: finalize_function_wrapper(fn) for name, fn in metrics.items()}

    # Setup pointwise evaluation
    def evaluate_data_point(dp: Data) -> dict[str, Array]:
        pred = {**pred_fn(dp["input"]), "target": dp["target"]}
        return finalize_functions(functions=metrics, results={}, aux=None, **pred)

    # Evaluate metrics
    evaluated_metrics = lmap(
        evaluate_data_point, data, batch_size=kwargs.get("lmap_eval_metrics", "data")
    )
    return {metric: apply(evaluated_metrics[metric]) for metric in evaluated_metrics}
