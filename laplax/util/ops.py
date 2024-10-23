"""Contains operations for flexible/adaptive compute."""

import os

import jax
import jax.numpy as jnp

# -------------------------------------------------------------------------
# Default values
# -------------------------------------------------------------------------

LAPLAX_PARALLELISM = 32
LAPLAX_DTYPE = "float32"
LAPLAX_PRECOMPUTE_LIST = "True"

# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------


def str_to_bool(value: str) -> bool:
    """Turning a str to bool."""
    if value not in {"True", "False"}:
        msg = "Invalid string representation of boolean value provided."
        raise ValueError(msg)
    return value == "True"


# -------------------------------------------------------------------------
# Adaptive operations
# -------------------------------------------------------------------------


def lmap(*args, **kwargs):  # noqa: ANN002
    batch_size = int(os.getenv("LAPLAX_PARALLELISM", LAPLAX_PARALLELISM))
    return jax.lax.map(*args, **kwargs, batch_size=batch_size)


def laplax_dtype(*args, **kwargs):  # noqa: ANN002
    del args, kwargs
    dtype = str(os.getenv("LAPLAX_DTYPE", LAPLAX_DTYPE))
    return jnp.dtype(dtype)


def precompute_list(func, _list):
    option = str_to_bool(os.getenv("LAPLAX_PRECOMPUTE_LIST", LAPLAX_PRECOMPUTE_LIST))

    if option:
        precompute_list = lmap(func, _list)

        def get_element(i):
            return precompute_list[i]

        return get_element

    return func
