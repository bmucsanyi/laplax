"""Generalized Gauss-Newton matrix-vector product and loss hessian."""

from collections.abc import Callable

import jax

from laplax.curv.hessian import hvp
from laplax.enums import LossFn
from laplax.types import (
    Array,
    Data,
    Float,
    InputArray,
    ModelFn,
    Num,
    Params,
    PredArray,
    TargetArray,
)
from laplax.util.ops import lmap
from laplax.util.tree import mul

# ---------------------------------------------------------------------
# Loss Hessian
# ---------------------------------------------------------------------


def _cross_entropy_hessian_mv(
    jv: PredArray, pred: PredArray, **kwargs
) -> Num[Array, "..."]:
    del kwargs
    prob = jax.nn.softmax(pred)
    off_diag_jv = prob * (prob.reshape(1, -1) @ jv)
    diag_jv = prob * jv
    return diag_jv - off_diag_jv


def _mse_hessian_mv(jv: PredArray, **kwargs) -> PredArray:
    del kwargs
    return 2 * jv


def create_loss_hessian_mv(
    loss_fn: LossFn | Callable[[PredArray, TargetArray], Num[Array, "..."]],
) -> Callable:
    """Return the Hessian-vector product for a given loss function.

    Create a function to compute the Hessian-vector product for a specified loss
    function.

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
    if loss_fn == LossFn.CROSSENTROPY:
        return _cross_entropy_hessian_mv

    if loss_fn == LossFn.MSE:
        return _mse_hessian_mv

    if isinstance(loss_fn, Callable):

        def custom_hessian_mv(
            jv: PredArray, pred: PredArray, target: TargetArray, **kwargs
        ) -> Num[Array, "..."]:
            del kwargs

            def loss_fn_local(p):
                return loss_fn(p, target)

            return hvp(loss_fn_local, pred, jv)

        return custom_hessian_mv

    msg = "Unsupported loss function provided."
    raise ValueError(msg)


# -----------------------------------------------------------------------------------
# GGN Matrix-vector product factories
# -----------------------------------------------------------------------------------


def create_ggn_mv_without_data(
    model_fn: ModelFn,
    params: Params,
    loss_fn: LossFn | Callable,
    factor: Float = 1.0,
    **kwargs,
) -> Callable[[Params, Data], Params]:
    """GGN-mv function without hardcoded data batch.

    Create a GGN-mv function that computes the Generalized Gauss-Newton (GGN) matrix-
    vector product without hardcoding the dataset.

    The formula for the GGN-mv is given by:

    $$ J_p^T H_L J_p v $$

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
        factor (float):
            Factor to scale the GGN-mv by.
        **kwargs:
            - lmap_ggn_mv (int, optional): Chunk size for iterating over the data.

    Returns:
        Callable:
            A function that takes a vector and a batch of data, then computes the GGN-mv
            for the specified model and loss function.

    Note:
        Function assumes that data has a batch dimension.
    """

    def _jvp_fn(
        params: Params, input: InputArray, vec: Params
    ) -> tuple[PredArray, PredArray]:
        def _local_model_fn(p):
            return model_fn(input=input, params=p)

        return jax.jvp(_local_model_fn, (params,), (vec,))

    def _vjp_fn(
        params: Params, input: InputArray
    ) -> tuple[PredArray, Callable[[PredArray], Params]]:
        def _local_model_fn(p):
            return model_fn(input=input, params=p)

        pred, vjp_fn = jax.vjp(_local_model_fn, params)
        return pred, lambda v: vjp_fn(v)[0]

    loss_hessian_mv = create_loss_hessian_mv(loss_fn)

    def _mv_ggn_ptw(input: InputArray, target: TargetArray, vec: Params) -> Params:
        """Calculate JT_p H_L J_p v for a single data point."""
        pred, jv = _jvp_fn(params, input, vec)
        hjv = loss_hessian_mv(jv, pred=pred, target=target)
        gv = _vjp_fn(params, input)[1](hjv)
        return gv

    def mv_ggn(vec: Params, data: Data) -> Params:
        def _mv_ggn_ptw_w_vec(dp: Data) -> Params:
            input, target = (
                dp["input"],
                dp["target"],
            )  # TODO(2bys): Do we want this constrain?
            return _mv_ggn_ptw(input, target, vec)

        return mul(
            factor,
            jax.lax.psum(
                lmap(
                    _mv_ggn_ptw_w_vec,
                    data,
                    batch_size=kwargs.get("lmap_ggn_mv", "data"),
                ),
                axis_name=0,
            ),
        )  # TODO(any): Should we handle the case factor=1. as a identity map?
        # (Possibly in util.tree.mul directly.)

    return mv_ggn


def create_ggn_mv(
    model_fn: ModelFn,
    params: Params,
    data: Data,
    loss_fn: LossFn | Callable,
    factor: Float = 1.0,
    **kwargs,
) -> Callable[[Params], Params]:
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
        factor (float):
            Factor to scale the GGN-mv by.
        **kwargs:
            - lmap_ggn_mv (int, optional): Chunk size for iterating over the data.

    Returns:
        Callable:
            A function that takes a vector and computes the GGN-mv for the specified
            model
            and dataset.
    """
    ggn_mv = create_ggn_mv_without_data(
        model_fn=model_fn, params=params, loss_fn=loss_fn, factor=factor, **kwargs
    )

    def wrapped_ggn_mv(vec: Params) -> Params:
        return ggn_mv(vec, data)

    return wrapped_ggn_mv
