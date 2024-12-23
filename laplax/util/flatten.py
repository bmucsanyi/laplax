"""Operations for flattening PyTrees into arrays."""

import math
from collections.abc import Generator

import jax
import jax.numpy as jnp

from laplax.types import Any, Array, Callable, PyTree
from laplax.util.utils import identity

# ---------------------------------------------------------------
# Flattening utilities
# ---------------------------------------------------------------


def cumsum(seq: Generator) -> list[int]:
    """Takes a sequence and returns the cumsum sequence."""
    total = 0
    return [total := total + ele for ele in seq]


def create_pytree_flattener(
    tree: PyTree,
) -> tuple[Callable[[PyTree], Array], Callable[[Array], PyTree]]:
    """Create flatten and unflatten functions for a one-dimensional (vector) PyTree."""

    def flatten(tree: PyTree) -> jax.Array:
        flat, _ = jax.tree.flatten(tree)
        return jnp.concatenate([leaf.ravel() for leaf in flat])

    # Get shapes and tree def for unflattening
    flat, tree_def = jax.tree.flatten(tree)
    all_shapes = [leaf.shape for leaf in flat]

    def unflatten(arr: Array) -> PyTree:
        flat_vector_split = jnp.split(
            arr, cumsum(math.prod(sh) for sh in all_shapes)[:-1]
        )
        return jax.tree.unflatten(
            tree_def,
            [
                a.reshape(sh)
                for a, sh in zip(flat_vector_split, all_shapes, strict=True)
            ],
        )

    return flatten, unflatten


def create_partial_pytree_flattener(
    tree: PyTree,
) -> tuple[Callable[[PyTree], Array], Callable[[Array], PyTree]]:
    """Create flatten and unflatten functions for partial PyTree arrays.

    Assumes an PyTree representing an array, where in each leaf the last
    dimension gives the column index, while the remaining might need to be
    flattened.
    """

    def flatten(tree: PyTree) -> jax.Array:
        flat, _ = jax.tree_util.tree_flatten(tree)
        return jnp.concatenate(
            [leaf.reshape(-1, leaf.shape[-1]) for leaf in flat], axis=0
        )

    # Get shapes and tree def for unflattening
    flat, tree_def = jax.tree_util.tree_flatten(tree)
    all_shapes = [leaf.shape for leaf in flat]

    def unflatten(arr: jax.Array) -> PyTree:
        flat_vector_split = jnp.split(
            arr, cumsum(math.prod(sh[1:]) for sh in all_shapes)[:-1], axis=1
        )  # Ignore row indices in shape.
        return jax.tree_util.tree_unflatten(
            tree_def,
            [
                flat_vector_split[i].reshape(all_shapes[i])
                for i in range(len(flat_vector_split))
            ],
        )

    return flatten, unflatten


def unravel_array_into_pytree(pytree: PyTree, axis: int, arr: Array) -> PyTree:
    """Unravel an array into a PyTree with a given structure.

    Args:
        pytree: The pytree that provides the structure.
        axis: The parameter axis is either -1, 0, or 1.  It controls the
          resulting shapes.
        example: If specified, cast the components to the matching dtype/weak_type,
          or else use the pytree leaf type if example is None.
        arr: The array to be unraveled.

    Reference: Following the implementation in jax._src.api._unravel_array_into_pytree
    """
    leaves, treedef = jax.tree.flatten(pytree)
    axis %= arr.ndim
    shapes = [arr.shape[:axis] + l.shape + arr.shape[axis + 1 :] for l in leaves]
    parts = jnp.split(arr, cumsum(math.prod(leaf.shape) for leaf in leaves[:-1]), axis)
    reshaped_parts = [x.reshape(shape) for x, shape in zip(parts, shapes, strict=True)]

    return jax.tree.unflatten(treedef, reshaped_parts)


def wrap_function(
    fn: Callable,
    input_fn: Callable | None = None,
    output_fn: Callable | None = None,
    argnums: int = 0,
) -> Callable:
    def wrapper(*args, **kwargs) -> Any:
        # Use the identity function if input_fn or output_fn is None
        effective_input_fn = input_fn or identity
        effective_output_fn = output_fn or identity

        # Call the original function on transformed input
        transformed_args = (
            *args[:argnums],
            effective_input_fn(args[argnums]),
            *args[argnums + 1 :],
        )
        result = fn(*transformed_args, **kwargs)

        # Apply the output transformation function
        return effective_output_fn(result)

    return wrapper


def wrap_factory(
    factory: Callable,
    input_fn: Callable | None = None,
    output_fn: Callable | None = None,
) -> Callable:
    def wrapped_factory(*args, **kwargs) -> Callable:
        fn = factory(*args, **kwargs)
        return wrap_function(fn, input_fn, output_fn)

    return wrapped_factory
