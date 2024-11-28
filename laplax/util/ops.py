"""Contains operations for flexible/adaptive compute."""

import operator
import os

import jax
import jax.numpy as jnp

# -------------------------------------------------------------------------
# Default values
# -------------------------------------------------------------------------

DEFAULT_PARALLELISM = 32
DEFAULT_DTYPE = "float32"
DEFAULT_PRECOMPUTE_LIST = "True"

# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------


def str_to_bool(value: str) -> bool:
    """Converts a string to a boolean."""
    valid_values = {"True": True, "False": False}
    if value not in valid_values:
        msg = "Invalid string representation of a boolean value."
        raise ValueError(msg)
    return valid_values[value]


def get_env_value(key: str, default: str) -> str:
    """Fetches the environment variable or returns the default."""
    return os.getenv(key, default)


def get_env_int(key: str, default: int) -> int:
    """Fetches an environment variable as an integer."""
    return int(get_env_value(key, str(default)))


def get_env_bool(key: str, default: str) -> bool:
    """Fetches an environment variable as a boolean."""
    return str_to_bool(get_env_value(key, default))


# -------------------------------------------------------------------------
# Adaptive operations
# -------------------------------------------------------------------------


def lmap(func, data, batch_size: int | str | None = None):
    """Support for `jax.lax.map` with flexible batch sizes.

    Args:
        func: The function to map over the data.
        data: The input data for mapping.
        batch_size: Batch size configuration, either:
            - None: Uses the default batch size.
            - str: Determines batch size from an environment variable.
            - int: Specifies the batch size directly.
    """
    if isinstance(batch_size, str):
        batch_size = get_env_int(
            f"LAPLAX_PARALLELISM_{batch_size.upper()}", DEFAULT_PARALLELISM
        )
    elif batch_size is None:
        batch_size = get_env_int("LAPLAX_PARALLELISM", DEFAULT_PARALLELISM)

    return jax.lax.map(func, data, batch_size=batch_size)


def laplax_dtype():
    """Returns the dtype specified by the environment or the default dtype."""
    dtype = get_env_value("LAPLAX_DTYPE", DEFAULT_DTYPE)
    return jnp.dtype(dtype)


def precompute_list(func, items, option: str | bool | None = None):  # noqa: FBT001
    """Precomputes a list of operations or returns the original function.

    Args:
        func: The function to apply to the items.
        items: A list of items to process.
        option: Precomputation control, either:
            - None: Uses the default precompute setting.
            - str: Determines the setting from an environment variable.
            - bool: Directly specifies whether to precompute.

    Returns:
        A function to retrieve precomputed elements by index, or `func`.
    """
    if isinstance(option, str):
        option = get_env_bool(
            f"LAPLAX_PRECOMPUTE_LIST_{option}", DEFAULT_PRECOMPUTE_LIST
        )
    elif option is None:
        option = get_env_bool("LAPLAX_PRECOMPUTE_LIST", DEFAULT_PRECOMPUTE_LIST)

    if option:
        precomputed = [func(item) for item in items]

        def get_element(index: int):
            return jax.tree_map(operator.itemgetter(index), precomputed)

        return get_element

    return func
