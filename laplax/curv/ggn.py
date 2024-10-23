"""Generalized Gauss-Newton matrix-vector product and loss hessian."""

from collections.abc import Callable

import jax

from laplax.curv.hessian import hvp
from laplax.util.ops import lmap

# ---------------------------------------------------------------------
# Loss Hessian
# ---------------------------------------------------------------------


def loss_hessian_mv_factory(loss_fn: str | Callable) -> Callable:
    """Return the Hessian-vector product for a given loss function.

    Args:
        loss_fn: loss function, either regression, cross_entropy, or callable.

    Returns:
        loss_hessian_mv (callable): Function taking a vector and returning the
            Hessian-vector product.
    """
    if loss_fn == "cross_entropy":

        def loss_hessian_mv(jv, *, pred, **kwargs):
            del kwargs
            prob = jax.nn.softmax(pred)
            off_diag_jv = prob * (prob.reshape(1, -1) @ jv)
            diag_jv = prob * jv
            return diag_jv - off_diag_jv

    elif loss_fn in {"regression", "mse"}:

        def loss_hessian_mv(jv, **kwargs):
            del kwargs
            return 2 * jv

    elif isinstance(loss_fn, Callable):

        def loss_hessian_mv(jv, pred, target, **kwargs):
            del kwargs

            def loss_fn_local(p):
                return loss_fn(p, target)

            return hvp(loss_fn_local, pred, jv)

    return loss_hessian_mv


def ggn_mv_factory(
    model_fn: Callable,
    params: dict,
    data: tuple[jax.Array, jax.Array],
    loss_fn: str | Callable,
) -> Callable:
    """Return a GGN-mv function for a given model, data, and loss function.

    Args:
        model_fn: Forward pass taking arguments params and input.
        params: Model parameters for model_fn.
        data: (input, target) tuple.
        loss_fn: loss function, either regression, cross_entropy, or callable.

    Returns:
        ggn_mv (callable): Function taking a vector and returning the GGN-mv.
    """

    def jvp_fn(params, input, vec):
        return jax.jvp(lambda p: model_fn(params=p, input=input), (params,), (vec,))

    def vjp_fn(params, input):
        return jax.vjp(lambda p: model_fn(params=p, input=input), params)

    loss_hessian_mv = loss_hessian_mv_factory(loss_fn)

    def mv_ggn_ptw(input, target, vec):
        """Calculate JT_p H_L J_p v for a single data point."""
        pred, jv = jvp_fn(params, input, vec)
        hjv = loss_hessian_mv(jv, pred, target)
        gv = vjp_fn(params, input)[1](hjv)[0]
        return gv

    def mv_ggn(vec):
        def mv_ggn_ptw_w_vec(dp):
            input, target = (
                dp["input"],
                dp["target"],
            )  # TODO(2bys): Do we want this constrain?
            return mv_ggn_ptw(input, target, vec)

        return jax.lax.psum(lmap(mv_ggn_ptw_w_vec, data), axis_name=0)

    return mv_ggn
