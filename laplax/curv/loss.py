"""This file contains the loss criterions available for curvature estimation."""

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------
# Cross-entropy loss (Classification)
# ---------------------------------------------------------------------


@jax.jit
def _set_diagonal(array: jax.Array, value: float) -> jax.Array:
    # Create an index array for the diagonal
    diag_indices = jnp.arange(array.shape[0])
    # Update the diagonal elements
    res = array.at[diag_indices, diag_indices].set(value)

    return res


def get_cross_entropy_loss_hessian(logit: jax.Array) -> jax.Array:
    """Calculate the Hessian matrix for cross-entropy loss.

    Args:
        logit: The input logits.

    Returns:
        The Hessian matrix.
    """
    prob = jax.nn.softmax(logit)
    off_diag_terms = -prob.reshape(-1, 1) @ prob.reshape(1, -1)
    hessian = _set_diagonal(off_diag_terms, prob * (1 - prob))

    return hessian


# ---------------------------------------------------------------------
# Mean Squared Error (Regression)
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# General Loss Hessian interface
# ---------------------------------------------------------------------
