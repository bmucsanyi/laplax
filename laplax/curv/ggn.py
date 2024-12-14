"""Generalized Gauss-Newton matrix-vector product and loss hessian."""

from collections.abc import Callable
from functools import partial

import jax

from laplax.curv.hessian import hvp
from laplax.util.ops import lmap

# ---------------------------------------------------------------------
# Loss Hessian
# ---------------------------------------------------------------------


def create_loss_hessian_mv(loss_fn: str | Callable) -> Callable:
    """Return the Hessian-vector product for a given loss function.

    Create a function to compute the Hessian-vector product for a specified loss
    function.

    $$ 5 + 5 = 10 $$
    Args:
        loss_fn (str | Callable):
            The loss function for which the Hessian-vector product is computed.
            Can be:

            - `"cross_entropy"`: Computes the Hessian-vector product for cross-entropy
                loss.
            - `"regression"` or `"mse"`: Computes the Hessian-vector product for mean
                squared error.
            - A custom callable loss function that takes predictions and targets as
                inputs.

    Returns:
        Callable:
            A function `loss_hessian_mv(jv, *, pred, target, **kwargs)` that computes
            the Hessian-vector product for the given loss function. The parameters are:
            - `jv` (jax.Array): The vector to multiply with the Hessian.
            - `pred` (jax.Array): The model predictions.
            - `target` (jax.Array, optional): The target labels (required for custom
                loss functions).
            - `**kwargs`: Additional arguments, which are ignored.
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


# -----------------------------------------------------------------------------------
# GGN Matrix-vector product factories
# -----------------------------------------------------------------------------------


def create_ggn_mv_without_data(
    model_fn: Callable,
    params: dict,
    loss_fn: str | Callable,
    **kwargs,
) -> Callable:
    """GGN-mv function without hardcoded data batch.

    Create a GGN-mv function that computes the Generalized Gauss-Newton (GGN) matrix-
    vector product without hardcoding the dataset.

    Args:
        model_fn (Callable):
            Forward pass function of the model, which takes `params` and `input` as
            arguments.
        params (dict):
            Model parameters to be passed to `model_fn`.
        loss_fn (str | Callable):
            Loss function to be used. Can be a string (e.g., "regression",
            "cross_entropy") or
            a callable that computes the loss.
        **kwargs:
            - lmap_ggn_mv (int, optional): Chunk size for iterating over the data.

    Returns:
        Callable:
            A function that takes a vector and a batch of data, then computes the GGN-mv
            for the specified model and loss function.
    """

    def jvp_fn(params, input, vec):
        return jax.jvp(lambda p: model_fn(params=p, input=input), (params,), (vec,))

    def vjp_fn(params, input):
        return jax.vjp(lambda p: model_fn(params=p, input=input), params)

    loss_hessian_mv = create_loss_hessian_mv(loss_fn)

    def mv_ggn_ptw(input, target, vec):
        """Calculate JT_p H_L J_p v for a single data point."""
        pred, jv = jvp_fn(params, input, vec)
        hjv = loss_hessian_mv(jv, pred=pred, target=target)
        gv = vjp_fn(params, input)[1](hjv)[0]
        return gv

    def mv_ggn(vec, data):
        def mv_ggn_ptw_w_vec(dp):
            input, target = (
                dp["input"],
                dp["target"],
            )  # TODO(2bys): Do we want this constrain?
            return mv_ggn_ptw(input, target, vec)

        return jax.lax.psum(
            lmap(mv_ggn_ptw_w_vec, data, batch_size=kwargs.get("lmap_ggn_mv", "data")),
            axis_name=0,
        )

    return mv_ggn


def create_ggn_mv(
    model_fn: Callable,
    params: dict,
    data: tuple[jax.Array, jax.Array],
    loss_fn: str | Callable,
    **kwargs,
) -> Callable:
    """GGN-mv factory function with hardcoded data batch.

    Create a GGN-mv function that computes the Generalized Gauss-Newton (GGN) matrix-
    vector product for a given model, dataset, and loss function.

    Args:
        model_fn (Callable):
            Forward pass function of the model, which takes `params` and `input` as
            arguments.
        params (dict):
            Model parameters to be passed to `model_fn`.
        data (tuple[jax.Array, jax.Array]):
            A tuple containing the input and target data.
        loss_fn (str | Callable):
            Loss function to be used. Can be a string (e.g., "regression",
            "cross_entropy") or a callable that computes the loss.
        **kwargs:
            - lmap_ggn_mv (int, optional): Chunk size for iterating over the data.

    Returns:
        Callable:
            A function that takes a vector and computes the GGN-mv for the specified
            model
            and dataset.
    """
    mv_ggn = create_ggn_mv_without_data(
        model_fn=model_fn, params=params, loss_fn=loss_fn, **kwargs
    )
    return partial(mv_ggn, data=data)
