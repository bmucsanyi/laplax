"""Utilities for handling DataLoaders/Iterables instead of single batches."""

import operator

import jax
import jax.numpy as jnp

from laplax.types import Any, Array, Callable, Data, Iterable, PyTree
from laplax.util.tree import add

# ------------------------------------------------------------------------
#  Data transformations
# ------------------------------------------------------------------------


def input_target_split(batch: tuple[Array, Array]) -> Data:
    return {"input": batch[0], "target": batch[1]}


# ------------------------------------------------------------------------
#  Reduction functions
# ------------------------------------------------------------------------


def reduce_add(
    res_new: Any, state: None | Any = None, *, keepdims: bool = True, axis: int = 0
) -> tuple[Any, Any]:
    """Add reduction with accumulated sum in state."""
    summed = jax.tree_map(lambda x: jnp.sum(x, keepdims=keepdims, axis=axis), res_new)
    if state is None:
        return summed, summed
    new_state = add(state, summed)
    return new_state, new_state


def concat(tree1: PyTree, tree2: PyTree, axis: int = 0) -> PyTree:
    return jax.tree.map(
        lambda x, y: jax.numpy.concatenate([x, y], axis=axis), tree1, tree2
    )


def reduce_concat(
    res_new: Any, state: None | Any = None, *, axis: int = 0
) -> tuple[Any, Any]:
    """Concatenate with accumulated results in state."""
    if state is None:
        return res_new, res_new
    new_state = concat(state, res_new, axis=axis)
    return new_state, new_state


def reduce_online_mean(res_new: Any, state: None | tuple = None) -> tuple[Any, tuple]:
    """Online mean with (count, running_sum) as state to avoid storing means."""
    batch_size = jax.tree_map(lambda x: x.shape[0] if x.ndim > 0 else 1, res_new)
    batch_sum = jax.tree_map(
        lambda x: jnp.sum(x, axis=0) if x.ndim > 0 else jnp.sum(x),
        res_new,
    )

    if state is None:
        return jax.tree_map(operator.truediv, batch_sum, batch_size), (
            batch_size,
            batch_sum,
        )

    old_count, old_sum = state
    total_count = jax.tree_map(operator.add, old_count, batch_size)
    new_sum = add(old_sum, batch_sum)

    current_mean = jax.tree_map(operator.truediv, new_sum, total_count)

    return current_mean, (total_count, new_sum)


# ------------------------------------------------------------------------
#  Core batch processing logic
# ------------------------------------------------------------------------


def process_batches(
    function: Callable,
    data_loader: Iterable,
    transform: Callable,
    reduce: Callable,
    *args,
    **kwargs,
) -> Any:
    """Core batch processing logic shared between wrapper implementations."""
    state = None
    result = None
    for batch in data_loader:
        result = function(*args, data=transform(batch), **kwargs)
        result, state = reduce(result, state)
    if result is None:
        msg = "Data loader was empty"
        raise ValueError(msg)
    return result


# ------------------------------------------------------------------------
#  Wrapper functions
# ------------------------------------------------------------------------


def execute_with_data_loader(
    function: Callable,
    data_loader: Iterable,
    transform: Callable = input_target_split,
    reduce: Callable = reduce_online_mean,
    *,
    jit: bool = False,
    **kwargs,
) -> Any:
    """Direct execution of batch processing."""
    fn = jax.jit(function) if jit else function
    return process_batches(fn, data_loader, transform, reduce, **kwargs)


def wrap_function_with_data_loader(
    function: Callable,
    data_loader: Iterable,
    transform: Callable = input_target_split,
    reduce: Callable = reduce_online_mean,
    *,
    jit: bool = False,
) -> Callable:
    """Returns a function that processes batches."""
    fn = jax.jit(function) if jit else function

    def wrapped(*args, **kwargs):
        return process_batches(fn, data_loader, transform, reduce, *args, **kwargs)

    return wrapped
