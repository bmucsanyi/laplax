"""Matrix-free array operations for matrix-vector products."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

from laplax import util
from laplax.types import Array, Layout, PyTree
from laplax.util.flatten import unravel_array_into_pytree
from laplax.util.ops import lmap
from laplax.util.tree import (
    basis_vector_from_index,
    eye_like,
    get_size,
)


def diagonal(mv: Callable | jnp.ndarray, layout: Layout | None = None) -> Array:
    """Return the diagonal of a PyTree-based matrix-vector-product.

    Args:
        mv (Callable | jax.Array): Matrix-vector product function.
        layout (int | PyTree | None): Specifies the layout of the matrix:
            - int: The size of the matrix.
            - PyTree: The structure for generating basis vectors.

    Returns:
        jax.Array: Diagonal of the matrix-free matrix.
    """
    if isinstance(mv, Callable) and not isinstance(layout, Layout):
        msg = "Either size or tree needs to be present."
        raise TypeError(msg)

    if isinstance(mv, jnp.ndarray):
        return jnp.diag(mv)

    # Define basis vector generator based on layout type
    if isinstance(layout, int):  # Integer layout defines size
        size = layout

        @jax.jit
        def get_basis_vec(idx: int) -> jax.Array:
            zero_vec = jnp.zeros(size)
            return zero_vec.at[idx].set(1.0)

    else:  # PyTree layout
        size = get_size(layout)

        @jax.jit
        def get_basis_vec(idx: int) -> PyTree:
            return basis_vector_from_index(idx, layout)

    # Compute the diagonal using basis vectors
    return jnp.stack([
        util.tree.tree_vec_get(mv(get_basis_vec(i)), i) for i in range(size)
    ])


def todense(mv: Callable, layout: Layout, **kwargs) -> Array:
    """Return a dense matrix representation of a matrix-vector product function.

    Args:
        mv (Callable): Matrix-vector product function.
        layout (dict | int | None): Specifies the structure of the mv input:
            - int: Specifies the size of the input dimension.
            - PyTree: Specifies the input structure for `mv`.
            - None: Assumes default identity-like structure.
        **kwargs: Define additional key word arguments
            - lmap_dense: Define batch size for densing mv.


    Returns:
        jax.Array: Dense matrix representation of the input matrix-vector product
            function.
    """
    # Create the identity-like basis based on `layout`
    if isinstance(layout, int):
        identity = jnp.eye(layout)
    elif isinstance(layout, PyTree):
        identity = eye_like(layout)
    else:
        msg = "`layout` must be an integer or a PyTree structure."
        raise TypeError(msg)

    return jax.tree.map(
        jnp.transpose, lmap(mv, identity, batch_size=kwargs.get("lmap_dense", "mv"))
    )  # Lmap shares along the first axis (rows instead of columns).


def todensetree(mv: Callable, layout: PyTree, **kwargs) -> PyTree:
    """Return a PyTree-to-PyTree representation of a matrix-vector product function.

    Args:
        mv (Callable): Matrix-vector product function.
        layout (PyTree): Specifies the structure of the input PyTree for `mv`.
        **kwargs: Define additional key word arguments
            - lmap_dense_tree: Define batch size for densing mv.

    Returns:
        PyTree: A PyTree representation of the dense matrix for the input-output
            relationship of `mv`.
    """
    # Create identity-like PyTree based on the layout
    identity = eye_like(layout)

    # Apply the matrix-vector product function to the identity-like PyTree
    mv_out = lmap(mv, identity, batch_size=kwargs.get("lmap_dense_tree", "mv"))

    # Convert the output into PyTree form
    return jax.tree.map(partial(unravel_array_into_pytree, layout, 0), mv_out)
