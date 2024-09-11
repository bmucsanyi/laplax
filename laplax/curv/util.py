# noqa: D100
from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def iterate_and_apply(
    dict_: dict[str, dict], func: Callable[[Any], Any]
) -> dict[str, dict]:
    # Applies function iteratively to all elements in dictionary
    for key, value in dict_.items():
        if isinstance(value, dict):
            dict_[key] = iterate_and_apply(value, func)
        else:
            dict_[key] = func(value)
    return dict_


def flatten_pytree(
    tree: Any,
) -> tuple[jax.Array, jax.tree_util.PyTreeDef, tuple[tuple]]:
    """Flattens a JAX PyTree into a 1D array."""
    leaves, tree_def = jax.tree.flatten(tree)
    flat_array = jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])
    shapes = tuple(leaf.shape for leaf in leaves)

    return flat_array, tree_def, shapes


def get_inflate_pytree_fn(
    tree_def: jax.tree_util.PyTreeDef, shapes: tuple[tuple]
) -> Callable[[jax.Array], Any]:
    # Convert arguments to tuples to be hashable (needed for jax.jit)
    sizes = tuple(np.prod(shape).item() for shape in shapes)
    split_indices = tuple(np.cumsum([0, *sizes[:-1]]).tolist())

    @partial(jax.jit, static_argnums=(1, 2, 3, 4))
    def inflate(
        flat_array: jax.Array,
        tree_def: jax.tree_util.PyTreeDef,
        split_indices: tuple[int],
        sizes: tuple[int],
        shapes: tuple[tuple],
    ):
        leaves = []
        for split_ind, size, shape in zip(split_indices, sizes, shapes, strict=False):
            leaves.append(
                jax.lax.dynamic_slice(flat_array, (split_ind,), (size,)).reshape(shape)
            )
        return jax.tree.unflatten(tree_def, leaves)

    return partial(
        inflate,
        tree_def=tree_def,
        split_indices=split_indices,
        sizes=sizes,
        shapes=shapes,
    )


def identity(x):
    return x
