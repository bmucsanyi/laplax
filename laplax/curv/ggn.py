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
    r"""Compute the Hessian-vector product for cross-entropy loss.

    This calculation uses the softmax probabilities of the predictions to compute the
    diagonal and off-diagonal components of the Hessian. The result is the difference
    between the diagonal contribution and the off-diagonal contribution of the Hessian.

    Mathematically, the Hessian-vector product is computed as:
    $H \cdot jv = \text{diag}(p) \cdot jv - p \cdot (p^\top \cdot jv), $ s
    where $p = \text{softmax}(\text{pred})$.

    Args:
        jv: Vector to multiply with the Hessian.
        pred: Model predictions (logits).
        **kwargs: Additional arguments (ignored).

    Returns:
        Hessian-vector product for cross-entropy loss.
    """
    del kwargs
    prob = jax.nn.softmax(pred)
    off_diag_jv = prob * (prob.reshape(1, -1) @ jv)
    diag_jv = prob * jv
    return diag_jv - off_diag_jv


def _mse_hessian_mv(jv: PredArray, **kwargs) -> PredArray:
    r"""Compute the Hessian-vector product for mean squared error loss.

    The Hessian of the mean squared error loss is a constant diagonal matrix with
    2 along the diagonal. Thus, the Hessian-vector product is simply 2 times the
    input vector.

    Mathematically:
    $H \cdot jv = 2 \cdot jv$.

    Args:
        jv: Vector to multiply with the Hessian.
        **kwargs: Additional arguments (ignored).

    Returns:
        Hessian-vector product for MSE loss.
    """
    del kwargs
    return 2 * jv


def create_loss_hessian_mv(
    loss_fn: LossFn | Callable[[PredArray, TargetArray], Num[Array, "..."]],
) -> Callable:
    r"""Create a function to compute the Hessian-vector product for a specified loss fn.

    For predefined loss functions like cross-entropy and mean squared error, the
    function computes their corresponding Hessian-vector products using efficient
    formulations. For custom loss functions, the Hessian-vector product is computed via
    automatic differentiation.

    Args:
        loss_fn: Loss function to compute the Hessian-vector product for. Supported
        options are:
            - "cross_entropy" for cross-entropy loss.
            - "mse" for mean squared error loss.
            - A custom callable loss function that takes predictions and targets.

    Returns:
        A function that computes the Hessian-vector product for the given loss function.
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
    r"""Create Generalized Gauss-Newton (GGN) matrix-vector productwithout fixed data.

    The GGN matrix is computed using the Jacobian of the model and the Hessian of the
    loss function. The resulting product is given by:
    $\text{factor} \cdot \sum_i J_i^\top H_{L, i} J_i \cdot v$
    where $J_i$ is the Jacobian of the model at data point $i$, $H_{L, i}$ is the
    Hessian of the loss, and $v$ is the vector.

    This function computes the above expression efficiently without hardcoding the
    dataset, making it suitable for distributed or batched computations.

    Args:
        model_fn: The model's forward pass function.
        params: Model parameters.
        loss_fn: Loss function to use for the GGN computation.
        factor: Scaling factor for the GGN computation.
        **kwargs: Additional arguments, including:
            - `lmap_ggn_mv`: Chunk size for iterating over data.

    Returns:
        A function that takes a vector and a batch of data, and computes the GGN
        matrix-vector product.

    Note:
        The function assumes that the data has a batch dimension.
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
    r"""Computes the Generalized Gauss-Newton (GGN) matrix-vector product with data.

    The GGN matrix is computed using the Jacobian of the model and the Hessian of the
    loss function. For a given dataset, the GGN matrix-vector product is computed as:
    $\text{factor} \sum_{i=1}^N J_i^\top H_{L, i} J_i \cdot v$
    where $J_i$ is the Jacobian of the model for the $i$-th data point, $H_{L, i}$ is
    the Hessian of the loss for the $i$-th data point, and $N$ is the number of data
    points.

    This function hardcodes the dataset, making it ideal for scenarios where the dataset
    remains fixed.

    Args:
        model_fn: The model's forward pass function.
        params: Model parameters.
        data: A batch of input and target data.
        loss_fn: Loss function to use for the GGN computation.
        factor: Scaling factor for the GGN computation.
        **kwargs: Additional arguments, including:
            - `lmap_ggn_mv`: Chunk size for iterating over data.

    Returns:
        A function that takes a vector and computes the GGN matrix-vector product for
        the given data.

    Note: The function assumes a batch dimension.
    """
    ggn_mv = create_ggn_mv_without_data(
        model_fn=model_fn, params=params, loss_fn=loss_fn, factor=factor, **kwargs
    )

    def wrapped_ggn_mv(vec: Params) -> Params:
        return ggn_mv(vec, data)

    return wrapped_ggn_mv
