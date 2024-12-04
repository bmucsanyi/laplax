"""Often used types defined in one place."""

from collections.abc import Callable, Iterable  # noqa: F401
from typing import Any  # noqa: F401

import jax
from jaxtyping import PRNGKeyArray, PyTree  # noqa: F401

# General types
KeyType = PRNGKeyArray
DType = jax.typing.DTypeLike
ShapeType = tuple[int, ...]
PyTreeDef = jax.tree_util.PyTreeDef
