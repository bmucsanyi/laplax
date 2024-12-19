"""Often used types defined in one place."""

from collections.abc import Callable, Iterable  # noqa: F401
from enum import StrEnum
from typing import Any  # noqa: F401

import jax
from jaxtyping import Array, Float, Num, PRNGKeyArray, PyTree  # noqa: F401

from laplax.enums import LossFn

# Package specific
Params = PyTree[Num[Array, "..."]]
ModelFn = Callable[[Num[Array, "..."]], Params]
CurvatureMV = Callable[[Params], Params]
PosteriorState = PyTree[Num[Array, "..."]]

Data = dict[str, Num[Array, "..."]]  # {"input": ..., "target": ...}
Layout = PyTree | int


# General types
KeyType = PRNGKeyArray
DType = jax.typing.DTypeLike
ShapeType = tuple[int, ...]
PyTreeDef = jax.tree_util.PyTreeDef
