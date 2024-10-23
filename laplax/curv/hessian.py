"""Full hessian estimation."""

import jax
import jax.numpy as jnp

from laplax.types import Callable


def model_hvp_factory(
    model_fn: Callable,
    params: dict,
    data: dict,
):
    """Set hvp-callable for model function and data.

    Args:
        model_fn: The function to estimate the Hessian of.
        function_input: The parameters at which to estimate the Hessian.
        data: The data samples to calculate the hessian from.

    Returns:
        hvp (Callable): The hessian-vector product.
    """
    # TODO(2bys): Implement this with similar flexibility.
    # grad_fn = lambda x: fn(x, data)
    # return jvp(grad(grad_fn), (params,), (new_v,))[1]
    # return # hvp
    pass


def hvp(func, primals, tangents):
    return jax.jvp(jax.grad(func), (primals,), (tangents,))[1]
