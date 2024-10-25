"""Often used types defined in one place."""

import collections.abc as cabc

import jax

# General types
Callable = cabc.Callable
KeyType = jax._src.prng.PRNGKeyArray  # noqa: SLF001
DType = jax.typing.DTypeLike
ShapeType = tuple[int, ...]
PyTree = dict  # TODO(2bys): Find the proper way of defining this type.
PyTreeDef = jax.tree_util.PyTreeDef
