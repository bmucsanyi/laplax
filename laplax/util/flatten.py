"""Operations for flattening PyTrees into arrays."""

import functools
import math
from functools import partial

import jax
import jax.numpy as jnp

from laplax.types import Callable, PyTree, PyTreeDef

# ---------------------------------------------------------------
# Flattening utilities
# ---------------------------------------------------------------


def cumsum(seq):
    """Takes a sequence and returns the cumsum sequence."""
    total = 0
    return [total := total + ele for ele in seq]


def flatten_pytree(
    tree: PyTree,
) -> tuple[jax.Array, PyTreeDef, tuple[tuple]]:
    """Flattens a JAX PyTree into a 1D array."""
    leaves, tree_def = jax.tree.flatten(tree)
    flat_array = jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])
    shapes = tuple(leaf.shape for leaf in leaves)

    return flat_array, tree_def, shapes


def get_inflate_pytree_fn(
    tree_def: jax.tree_util.PyTreeDef, shapes: tuple[tuple]
) -> Callable[[jax.Array], PyTree]:
    # Convert arguments to tuples to be hashable (needed for jax.jit)
    sizes = tuple(math.prod(shape).item() for shape in shapes)
    split_indices = tuple(cumsum([0, *sizes[:-1]]).tolist())

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


def flatten_hessian(hessian_pytree: PyTree, params_pytree: PyTree) -> jax.Array:
    """Flatten the Hessian matrix.

    Args:
        hessian_pytree: The Hessian matrix represented as a PyTree.
        params_pytree: The parameters represented as a PyTree.

    Returns:
        The flattened Hessian matrix.
    """
    # Tree flatten both hessian and params
    flatten_tree = jax.tree_util.tree_flatten(hessian_pytree)[0]
    flatten_params = jax.tree_util.tree_flatten(params_pytree)[0]

    # Concatenate hessian to tree
    n_parts = len(flatten_params)
    full_hessian = jnp.concatenate(
        [
            jnp.concatenate(
                [
                    arr.reshape(math.prod(p.shape), -1)
                    for arr in flatten_tree[i * n_parts : (i + 1) * n_parts]
                ],
                axis=1,
            )
            for i, p in enumerate(flatten_params)
        ],
        axis=0,
    )

    return full_hessian


def inflate_and_flatten(flatten_fn: callable, inflate_fn: callable, argnums: int = 0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):  # noqa: ANN002
            # Flatten the input to get the tree definition and shapes
            args[argnums] = inflate_fn(args[argnums])

            # Call the original function with the inflated input
            result = func(*args, **kwargs)

            # Flatten the output
            flat_output = flatten_fn(result)

            return flat_output

        return wrapper

    return decorator
