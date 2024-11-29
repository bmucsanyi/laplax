"""Full hessian estimation."""

import jax


def hvp(func, primals, tangents):
    return jax.jvp(jax.grad(func), (primals,), (tangents,))[1]
