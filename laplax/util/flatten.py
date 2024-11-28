"""Operations for flattening PyTrees into arrays."""

import functools
import math
from functools import partial

import jax
import jax.numpy as jnp

from laplax.types import Any, Callable, PyTree, PyTreeDef

# ---------------------------------------------------------------
# Flattening utilities
# ---------------------------------------------------------------


def cumsum(seq):
    """Takes a sequence and returns the cumsum sequence."""
    total = 0
    return [total := total + ele for ele in seq]


def create_pytree_flattener(tree: PyTree):
    """Create flatten and unflatten functions for a one-dimensional (vector) PyTree."""

    def flatten(tree: PyTree) -> jax.Array:
        flat, _ = jax.tree.flatten(tree)
        return jnp.concatenate([leaf.ravel() for leaf in flat])

    # Get shapes and tree def for unflattening
    flat, tree_def = jax.tree.flatten(tree)
    all_shapes = [leaf.shape for leaf in flat]

    def unflatten(arr: jax.Array) -> PyTree:
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


def create_partial_pytree_flattener(tree: PyTree):
    """Create flatten and unflatten functions for partial PyTree arrays.

    Assumes an PyTree representing an array, where in each leaf the first
    dimension gives the row index, while the remaining might need to be
    flattened.
    """

    def flatten(tree: PyTree) -> jax.Array:
        flat, _ = jax.tree_util.tree_flatten(tree)
        return jnp.concatenate(
            [leaf.reshape(leaf.shape[0], -1) for leaf in flat], axis=1
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


def identity(x: Any) -> Any:
    return x


def wrap_function(
    fn: Callable,
    input_fn: Callable | None = None,
    output_fn: Callable | None = None,
    argnums: int = 0,
) -> Callable:
    def wrapper(*args, **kwargs):
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
) -> any:
    def wrapped_factory(*args, **kwargs) -> Callable:  # noqa: ANN002
        fn = factory(*args, **kwargs)
        return wrap_function(fn, input_fn, output_fn)

    return wrapped_factory


def inflate_and_flatten(flatten_fn: Callable, inflate_fn: Callable, argnums: int = 0):
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


def unravel_array_into_pytree(pytree, axis, arr):
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


def flatten_pytree_to_array(tree):
    """Flatten a PyTree into a 1D array."""
    flat, _ = jax.tree.flatten(tree)
    return jnp.concatenate([x.ravel() for x in flat])
