"""General utility functions."""

import jax

from laplax.types import Any, Callable, Iterable
from laplax.util.tree import add


def identity(x: any) -> any:
    return x


def input_target_split(batch) -> dict[jax.Array, jax.Array]:
    return {"input": batch[0], "target": batch[1]}


def reduce_add(res_new: Any, res_old: Any | None = None) -> Any:
    return add(res_new, res_old) if res_old else res_new


def wrap_function_with_data_loader(
    function: Callable,
    data_loader: Iterable,
    transform: Callable = input_target_split,
    reduce: Callable = reduce_add,
    **function_kwargs,
):
    """Wraps a callable to process data in batches.

    Args:
        function (Callable):
            The function to wrap, which accepts `*args`, `**kwargs`, and a data
            argument containing a batch of transformed data.
            data_loader (Iterable):

        data_loader (Iterable):
            An iterable that yields batches of data to be processed.

        transform (Callable, optional):
            A function to preprocess each batch before passing it to `function`.
            Defaults to `input_target_split`, which formats batches into dictionaries
            with "input" and "target" keys.

        reduce (Callable, optional):
            A function to combine results from processing individual batches. Defaults
            to `reduce_add`, which sums results across batches.

        **function_kwargs:
            Additional arguments for configuring the wrapped function. If `"jit"=True`,
            JIT compilation is applied using `jax.jit`.

    Returns:
        Callable:
            Wrapped function.
    """

    def fn(*args, data, **kwargs):
        return function(*args, data=transform(data), **kwargs)

    if function_kwargs.get("jit", False):
        fn = jax.jit(fn)

    def wrapped_function(*args, **kwargs):
        res_old = None
        for batch in data_loader:
            res_new = fn(*args, data=batch, **kwargs)
            res_old = reduce(res_new, res_old)
        return res_old

    return wrapped_function
