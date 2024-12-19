"""Rosenbrock function and its sub-functions for testing purposes."""

import math
from functools import partial
from math import sqrt

import jax.numpy as jnp


def rosenbrock_first(x: jnp.ndarray, alpha: float):
    """Evaluate first sub-function of the Rosenbrock function.

    Args:
        x: A two-dimensional vector.
        alpha: The Rosenbrock function'ss parameter.

    Returns:
        A two-dimensional vector containing the evaluation.
    """
    assert x.ndim == 1
    assert math.prod(x.shape) == 2

    g0, g1 = 1 - x[0], sqrt(alpha) * (x[1] - x[0] ** 2)
    return jnp.stack([g0, g1])


def rosenbrock_last(g: jnp.ndarray):
    """Evaluate the last sub-function of the Rosenbrock function.

    Args:
        g: A two-dimensional vector containing the evaluation of the first sub-function.

    Returns:
        The evaluatoin of the last sub-function (a scalar).
    """
    assert g.ndim == 1
    assert math.prod(g.shape) == 2
    return jnp.sum(g**2)


def rosenbrock_fn(
    x: jnp.ndarray,
    alpha: float,
    *,
    return_first: bool = False,
) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate the Rosenbrock function.

    Args:
        x: A two-dimensional vector.
        alpha: The Rosenbrock function's parameter.
        return_first: Whether to return the evaluation of the
        first sub-function as well. Defaults to False.

    Returns:
        The evaluation of the Rosenbrock function.
        If return_first is True, a tuple containing the
        evaluation of the last sub-function and the first
        sub-function is returned.
    """
    assert x.ndim == 1
    assert math.prod(x.shape) == 2

    first = rosenbrock_first(x, alpha)
    last = rosenbrock_last(first)

    return last, first if return_first else last


class RosenbrockCase:
    """Rosenbrock function with manual hessian and ggn for testing purposes."""

    def __init__(self, x, alpha):
        self.x = x
        self.alpha = alpha

    @property
    def rosenbrock(self):
        return partial(rosenbrock_fn, alpha=self.alpha)

    @property
    def model_fn(self):
        return lambda input, params: rosenbrock_first(params, alpha=self.alpha)  # noqa: ARG005

    @property
    def loss_fn(self):
        return lambda pred, target: rosenbrock_last(pred)  # noqa: ARG005

    @property
    def hessian_manual(
        self,
    ) -> jnp.ndarray:
        x, alpha = self.x, self.alpha
        h00 = 2 + 12 * alpha * x[0] ** 2 - 4 * alpha * x[1]
        h01 = -4 * alpha * x[0]
        h10 = -4 * alpha * x[0]
        h11 = 2 * alpha
        return jnp.array([[h00, h01], [h10, h11]])

    @property
    def ggn_manual(
        self,
    ) -> jnp.ndarray:
        x, alpha = self.x, self.alpha
        h00 = 1 + 4 * alpha * x[0] ** 2
        h01 = -2 * alpha * x[0]
        h10 = -2 * alpha * x[0]
        h11 = alpha

        return 2 * jnp.array([[h00, h01], [h10, h11]])
