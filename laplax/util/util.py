"""General utility functions."""

import jax
import jax.numpy as jnp

from laplax.types import Any, Callable, Iterable, PyTree
from laplax.util.tree import add


def identity(x: any) -> any:
    return x


def input_target_split(batch) -> dict[jax.Array, jax.Array]:
    return {"input": batch[0], "target": batch[1]}


def reduce_axis_add(
    res_new: Any,
    res_old: None | Any = None,
    reduce_state: None | Any = None,
    axis=0,
) -> Any:
    res_new = jax.tree.map(
        lambda x: jax.numpy.sum(x, axis=axis, keepdims=True), res_new
    )
    if res_old:
        res_old = jax.tree.map(lambda x: jax.numpy.sum(x, axis=axis), res_old)
        return add(res_new, res_old), reduce_state

    return res_new, reduce_state


def reduce_add(
    res_new: Any,
    res_old: None | Any = None,
    reduce_state: None | Any = None,
) -> Any:
    if res_old:
        return add(res_new, res_old), reduce_state
    return res_new, reduce_state
    # return add(res_new, res_old), reduce_state if res_old else res_new, reduce_state


def concat(tree1, tree2, axis=0):
    return jax.tree.map(
        lambda x, y: jax.numpy.concatenate([x, y], axis=axis), tree1, tree2
    )


def reduce_concat(
    res_new: Any, res_old: None | Any = None, reduce_state: None | Any = None
) -> tuple[PyTree, Any]:
    if res_old:
        return concat(res_new, res_old), reduce_state
    return res_new, reduce_state


def reduce_online_mean(
    res_new: Any, res_old: None | Any = None, reduce_state: None | tuple = None
) -> tuple[PyTree, Any]:
    """Computes an online mean of batched results.

    Args:
        res_new: Current batch result.
        res_old: Previous accumulated mean.
        reduce_state: Tuple of (current count, previous mean).

    Returns:
        Tuple of updated mean and new reduce state.

    """
    batch_size = jax.tree_util.tree_map(
        lambda x: x.shape[0] if x.ndim > 0 else 1, res_new
    )

    if reduce_state is None:
        batch_mean = jax.tree_util.tree_map(
            lambda x: jnp.mean(x, axis=0) if x.ndim > 0 else jnp.mean(x),
            res_new,
        )
        return batch_mean, (batch_size, batch_mean)

    old_count, old_mean = reduce_state
    batch_mean = jax.tree_util.tree_map(
        lambda x: jnp.mean(x, axis=0) if x.ndim > 0 else jnp.mean(x), res_new
    )

    total_count = jax.tree_util.tree_map(lambda x, y: x + y, old_count, batch_size)

    new_mean = jax.tree_util.tree_map(
        lambda old_m, old_c, new_m, new_c: (old_m * old_c + new_m * new_c)
        / (old_c + new_c),
        old_mean,
        old_count,
        batch_mean,
        batch_size,
    )

    return new_mean, (total_count, new_mean)


def execute_with_data_loader(
    function: Callable,
    data_loader: Iterable,
    transform: Callable = input_target_split,
    reduce: Callable = reduce_online_mean,
    jit: bool = False,
    **kwargs,
):
    return wrap_function_with_data_loader(
        function=function,
        data_loader=data_loader,
        transform=transform,
        reduce=reduce,
        jit=jit,
    )(**kwargs)


def wrap_function_with_data_loader(
    function: Callable,
    data_loader: Iterable,
    transform: Callable = input_target_split,
    reduce: Callable = reduce_online_mean,
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
            A function to preprocess each batch before passing it to `function`. Defaults
            to `input_target_split`, which formats batches into dictionaries with "input"
            and "target" keys.

        reduce (Callable, optional):
            A function to combine results from processing individual batches. Defaults to
            `reduce_add`, which sums results across batches.

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
        reduce_state = kwargs.get("reduce_state")
        for batch in data_loader:
            res_new = fn(*args, data=batch, **kwargs)
            res_old, reduce_state = reduce(res_new, res_old, reduce_state=reduce_state)
        return res_old

    return wrapped_function


import jax
import jax.numpy as jnp

from laplax.types import Any, PyTree
from laplax.util.tree import add


# Simple Test
def test_reduce_online_mean():
    batch1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    batch2 = jnp.array([[5.0, 6.0], [7.0, 8.0]])

    mean, state = reduce_online_mean(batch1)
    print("After batch1:", mean)
    # Should be [2. 3.]

    mean, state = reduce_online_mean(batch2, reduce_state=state)
    print("After batch2:", mean)
    # Should be [4. 5.]


if __name__ == "__main__":
    test_reduce_online_mean()
