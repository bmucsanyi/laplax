"""Contains operations for flexible/adaptive compute."""

import operator
import os

import jax
import jax.numpy as jnp

from laplax.types import Any, Callable, DType, Iterable

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
    """Convert a string representation of a boolean to a boolean value.

    Args:
        value: A string representation of a boolean ("True" or "False").

    Returns:
        bool: The corresponding boolean value.

    Raises:
        ValueError: If the string does not represent a valid boolean value.
    """
    valid_values = {"True": True, "False": False}
    if value not in valid_values:
        msg = "Invalid string representation of a boolean value."
        raise ValueError(msg)
    return valid_values[value]


def get_env_value(key: str, default: str) -> str:
    """Fetch the value of an environment variable or return a default value.

    Args:
        key: The name of the environment variable.
        default: The default value to return if the variable is not set.

    Returns:
        str: The value of the environment variable or the default.
    """
    return os.getenv(key, default)


def get_env_int(key: str, default: int) -> int:
    """Fetch the value of an environment variable as an integer.

    Args:
        key: The name of the environment variable.
        default: The default integer value to return if the variable is not set.

    Returns:
        int: The value of the environment variable as an integer.
    """
    return int(get_env_value(key, str(default)))


def get_env_bool(key: str, default: str) -> bool:
    """Fetch the value of an environment variable as a boolean.

    Args:
        key: The name of the environment variable.
        default: The default string value ("True" or "False") if the variable is not
            set.

    Returns:
        bool: The value of the environment variable as a boolean.

    Raises:
        ValueError: If the default string is not a valid boolean representation.
    """
    return str_to_bool(get_env_value(key, default))


# -------------------------------------------------------------------------
# Adaptive operations
# -------------------------------------------------------------------------


def lmap(func: Callable, data: Iterable, batch_size: int | str | None = None) -> Any:
    """Apply a function over an iterable with support for batching.

    This function maps `func` over `data`, splitting the data into batches
    determined by the `batch_size`.

    Args:
        func: The function to apply to each element or batch of the data.
        data: The input iterable over which to map the function.
        batch_size: The batch size for processing. Options:
            - None: Use the default batch size specified by the environment.
            - str: Determine the batch size from an environment variable.
            - int: Specify the batch size directly.

    Returns:
        Any: The result of mapping `func` over `data`.
    """
    if isinstance(batch_size, str):
        batch_size = get_env_int(
            f"LAPLAX_PARALLELISM_{batch_size.upper()}", DEFAULT_PARALLELISM
        )
    elif batch_size is None:
        batch_size = get_env_int("LAPLAX_PARALLELISM", DEFAULT_PARALLELISM)

    return jax.lax.map(func, data, batch_size=batch_size)


def laplax_dtype() -> DType:
    """Get the data type (dtype) used by the library.

    This function retrieves the dtype specified by the "LAPLAX_DTYPE" environment
    variable or returns the default dtype.

    Returns:
        DType: The JAX-compatible dtype to use.
    """
    dtype = get_env_value("LAPLAX_DTYPE", DEFAULT_DTYPE)
    return jnp.dtype(dtype)


def precompute_list(
    func: Callable, items: Iterable, option: str | bool | None = None, **kwargs
) -> Callable:
    """Precompute results for a list of items or return the original function.

    If `option` is enabled, this function applies `func` to all items in `items`
    and stores the results for later retrieval. Otherwise, it returns `func` as-is.

    Args:
        func: The function to apply to each item in the list.
        items: An iterable of items to process.
        option: Determines whether to precompute results:
            - None: Use the default precompute setting.
            - str: Retrieve the setting from an environment variable.
            - bool: Specify directly whether to precompute.
        **kwargs: Additional keyword arguments, including:
            - lmap_precompute: Batch size for precomputing results.

    Returns:
        Callable: A function to retrieve precomputed elements by index, or the original
        `func` if precomputation is disabled.
    """
    if isinstance(option, str):
        option = get_env_bool(
            f"LAPLAX_PRECOMPUTE_LIST_{option}", DEFAULT_PRECOMPUTE_LIST
        )
    elif option is None:
        option = get_env_bool("LAPLAX_PRECOMPUTE_LIST", DEFAULT_PRECOMPUTE_LIST)

    if option:
        precomputed = lmap(
            func, items, batch_size=kwargs.get("lmap_precompute", "precompute")
        )

        def get_element(index: int):
            return jax.tree.map(operator.itemgetter(index), precomputed)

        return get_element

    return func
