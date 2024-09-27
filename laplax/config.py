import os  # noqa: D100

import jax
import jax.numpy as jnp 

# Default values for the environment variables.
# Parallelism for data evaluation.
# Set alternative via: os.environ["LAPLAX_PARALLELISM"] = "32"
LAPLAX_PARALLELISM = 32
# Set alternative via: os.environ["LAPLAX_DTYPE"] = "float64"
LAPLAX_DTYPE = 'float32'

def lmap(*args, **kwargs):  # noqa: ANN002
    batch_size = int(os.getenv("LAPLAX_PARALLELISM", LAPLAX_PARALLELISM))
    return jax.lax.map(*args, **kwargs, batch_size=batch_size)

def laplax_dtype(*args, **kwargs):
    dtype = str(os.getenv("LAPLAX_DTYPE", LAPLAX_DTYPE))
    return jnp.dtype(dtype)