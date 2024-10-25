"""Matrix-free array operations for matrix-vector products."""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from laplax.util.ops import lmap
from laplax.util.tree import basis_vector_from_index, eye_like


def diagonal(mv: Callable, size: int, tree: dict | None = None) -> jax.Array:
    """Return the diagonal of a PyTree-based matrix-vector-product.

    Args:
        mv (Callable): Matrix-vector product function.
        size (int): Size of the matrix-free matrix.
        tree: If mv operates on trees.

    Return:
        jax.Array: Diagonal of the matrix-free matrix.
    """
    if tree:

        @jax.jit
        def get_basis_vec(idx: int):
            return basis_vector_from_index(tree, idx)

    else:

        @jax.jit
        def get_basis_vec(idx: int):
            zero_vec = jnp.zeros(size)
            return zero_vec.at[idx].set(1.0)

    return jnp.stack([mv(get_basis_vec(i))[i] for i in range(size)])


def todense(mv: Callable, like: dict | int | None = None) -> jax.Array:
    """Return dense matrix for mv-product."""
    identity = jnp.eye(like) if isinstance(like, int) else eye_like(like)
    return lmap(mv, identity)


def array_to_mv(arr: jax.Array) -> Callable:
    def _mv(vec):
        return arr @ vec

    return _mv
