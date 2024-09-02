"""This file contains the additional functions to perform diagonal estimation."""

import jax
import jax.numpy as jnp


def get_ggn_diag(ggn: jax.Array) -> jax.Array:
    """Get the diagonal of the GGN matrix."""

    @jax.jit
    def get_canonical_basis_vector(index: int):
        zero_vec = jnp.zeros(ggn.shape[0], dtype=ggn.dtype)
        return zero_vec.at[index].set(1)

    ggn_mul = jax.jit(ggn.__matmul__)
    return jnp.stack([
        ggn_mul(get_canonical_basis_vector(index))[index]
        for index in range(ggn.shape[0])
    ])
