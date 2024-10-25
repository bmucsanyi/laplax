"""Relevant tree operations."""

import math
import operator
from functools import partial
from itertools import starmap

import jax
import jax.numpy as jnp

from laplax.util.ops import lmap

# ---------------------------------------------------------------
# Tree utilities
# ---------------------------------------------------------------


def get_size(tree):
    flat, _ = jax.tree_util.tree_flatten(tree)
    return sum(math.prod(arr.shape) for arr in flat)


# ---------------------------------------------------------------
# Create common arrays in given tree structure
# ---------------------------------------------------------------


def ones_like(tree):
    return jax.tree.map(jnp.ones_like, tree)


def zeros_like(tree):
    return jax.tree.map(jnp.zeros_like, tree)


def randn_like(key, tree):
    """Generates a White Noise PyTree in the form of given PyTree.

    Args:
        key : PRNGKey,
        pytree: A PyTree of arrays, whose structure and shape will be used.

    Returns:
        (PyTree) : A PyTree of random numbers with the same structure
            and shape as `pytree`.
    """
    # Split the key according to number of leaves
    keys = jax.random.split(key, len(jax.tree_util.tree_leaves(tree)))

    # Define a random number generator
    def generate_random_leaf(leaf, subkey):
        return jax.random.normal(subkey, shape=leaf.shape)

    # Apply the function to each leaf of the pytree.
    return jax.tree.map(generate_random_leaf, tree, keys)


def basis_vector_from_index(idx, tree):
    # Create a tree of zeros with the same structure
    zeros = zeros_like(tree)

    # Flatten the tree to get a list of arrays and the tree definition
    flat, tree_def = jax.tree_util.tree_flatten(zeros)

    # Compute the cumulative sizes of each array in the flat structure
    sizes = jnp.array([math.prod(arr.shape) for arr in flat])

    # Find the index of the array containing the idx
    cum_sizes = jnp.cumsum(sizes)
    k = jnp.searchsorted(cum_sizes, idx, side="right")

    # Compute the adjusted index within the identified array
    idx_corr = idx - jnp.where(k > 0, cum_sizes[k - 1], 0)

    # Define a function to update the k-th array with the basis vector
    def update_array(i, arr):
        return jax.lax.cond(
            i == k,
            lambda: arr.flatten().at[idx_corr].set(1.0).reshape(arr.shape),
            lambda: arr,
        )

    # Map the update function across the flattened list of arrays
    updated_flat = list(starmap(update_array, enumerate(flat)))

    # Reconstruct the tree with the updated flat structure
    return jax.tree_util.tree_unflatten(tree_def, updated_flat)


def eye_like(tree):
    n_ele = get_size(tree)
    return lmap(partial(basis_vector_from_index, tree=tree), jnp.arange(n_ele))


def slice(tree, a, b):
    return jax.tree.map(operator.itemgetter(slice(a, b)), tree)


# ---------------------------------------------------------------
# Tree operations
# ---------------------------------------------------------------


def add(tree1, tree2):
    return jax.tree.map(jnp.add, tree1, tree2)


def neg(tree):
    return jax.tree.map(jnp.negative, tree)


def sub(tree1, tree2):
    return add(tree1, neg(tree2))


def invert(tree):
    return jax.tree.map(jnp.invert, tree)


def mean(tree, **kwargs):
    return jax.tree.map(partial(jnp.mean, **kwargs), tree)


def std(tree, **kwargs):
    return jax.tree.map(partial(jnp.std, **kwargs), tree)


def cov(tree, **kwargs):
    return jax.tree.map(partial(jnp.cov, **kwargs), tree)


# ---------------------------------------------------------------
# For testing
# ---------------------------------------------------------------


def allclose(tree1, tree2):
    return jax.tree.all(jax.tree.map(jnp.allclose, tree1, tree2))
