"""Matrix-free array operations for matrix-vector products."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

from laplax.types import PyTree
from laplax.util.flatten import unravel_array_into_pytree
from laplax.util.ops import lmap
from laplax.util.tree import (
    basis_vector_from_index,
    eye_like,
    tree_matvec,
    tree_partialmatvec,
)


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


def todensetree(mv: Callable, tree: PyTree) -> PyTree:
    """Return pytree-pytree for mv-product."""
    identity = eye_like(tree)
    mv_out = lmap(mv, identity)
    return jax.tree.map(partial(unravel_array_into_pytree, tree, 0), mv_out)


def array_to_mv(
    arr: jax.Array,
    flatten: Callable | None = None,
    unflatten: Callable | None = None,
) -> Callable:
    def _mv(vec):
        if flatten:
            vec = flatten(vec)
        vec = arr @ vec
        return vec if unflatten is None else unflatten(vec)

    return _mv


def tree_to_mv(tree: PyTree) -> Callable:
    def _mv(vec):
        return tree_matvec(tree, vec)

    return _mv


def partialtree_to_mv(tree: PyTree) -> Callable:
    def _mv(vec):
        return tree_partialmatvec(tree, vec)

    return _mv
