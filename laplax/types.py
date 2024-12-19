"""All types defined in one place."""

from collections.abc import Callable, Iterable  # noqa: F401
from typing import Any  # noqa: F401

import jax
from jaxtyping import Array, Float, Num, PRNGKeyArray, PyTree  # noqa: F401

# Basic JAX types
KeyType = PRNGKeyArray
DType = jax.typing.DTypeLike
ShapeType = tuple[int, ...]
PyTreeDef = jax.tree_util.PyTreeDef

# Array types
InputArray = Num[Array, "..."]
PredArray = Num[Array, "..."]
TargetArray = Num[Array, "..."]
FlatParams = Num[Array, "P"]

# Parameter and model types
Params = PyTree[Num[Array, "..."]]
ModelFn = Callable[..., PredArray]  # [InputArray, Params]
CurvatureMV = Callable[[Params], Params]

# Data structures
Data = dict[str, Num[Array, "..."]]  # {"input": ..., "target": ...}
Layout = PyTree | int
PriorArguments = dict[str, Array]
PosteriorState = PyTree[Num[Array, "..."]]
