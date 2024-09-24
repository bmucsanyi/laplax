import os  # noqa: D100

import jax

# Default values for the environment variables.
# Parallelism for data evaluation.
# Set alternative via: os.environ["LAPLAX_PARALLELISM"] = "32"
LAPLAX_PARALLELISM = 32


def lmap(*args, **kwargs):  # noqa: ANN002
    batch_size = os.getenv("LAPLAX_PARALLELISM", LAPLAX_PARALLELISM)
    return jax.lax.map(*args, **kwargs, batch_size=batch_size)
