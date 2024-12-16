"""General utility functions."""

import jax


def identity(x: any) -> any:
    return x


def input_target_split(batch) -> dict[jax.Array, jax.Array]:
    return {"input": batch[0], "target": batch[1]}
