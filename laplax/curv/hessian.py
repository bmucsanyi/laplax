"""Hessian vector product for curvature estimation."""

import jax
import jax.numpy as jnp

from laplax.enums import LossFn
from laplax.types import (
    Array,
    Callable,
    Data,
    InputArray,
    ModelFn,
    Num,
    Params,
    PredArray,
    PyTree,
    TargetArray,
)


def hvp(func: Callable, primals: PyTree, tangents: PyTree) -> PyTree:
    r"""Compute the Hessian-vector product (HVP) for a given function.

    The Hessian-vector product is computed by differentiating the gradient of the
    function. This avoids explicitly constructing the Hessian matrix, making the
    computation efficient.

    Args:
        func: The scalar function for which the HVP is computed.
        primals: The point at which the gradient and Hessian are evaluated.
        tangents: The vector to multiply with the Hessian.

    Returns:
        The Hessian-vector product.
    """
    return jax.jvp(jax.grad(func), (primals,), (tangents,))[1]


def concatenate_model_and_loss_fn(
    model_fn: ModelFn,  # type: ignore[reportRedeclaration]
    loss_fn: LossFn | Callable | None = None,
    *,
    has_batch: bool = False,
) -> Callable[[InputArray, TargetArray, Params], Num[Array, "..."]]:
    r"""Combine a model function and a loss function into a single callable.

    This creates a new function that evaluates the model and applies the specified
    loss function. If `has_batch` is `True`, the model function is vectorized over
    the batch dimension using `jax.vmap`.

    Mathematically, the combined function computes:
    $$L(x, y, \theta) = \text{loss}(f(x, \theta), y)$$
    where $f$ is the model function, $\theta$ are the model parameters, $x$ is the
    input, and $y$ is the target.

    Args:
        model_fn: The model function to evaluate.
        loss_fn: The loss function to apply. Supported options are:
            - `LossFn.MSE` for mean squared error.
            - `LossFn.CROSSENTROPY` for cross-entropy loss.
            - A custom callable loss function.
        has_batch: Whether the model function should be vectorized over the batch.

    Returns:
        A combined function that computes the loss for given inputs, targets, and
        parameters.
    """
    if has_batch:
        model_fn = jax.vmap(model_fn, in_axes=(0, None))

    if loss_fn == LossFn.MSE:

        def loss_wrapper(
            input: InputArray, target: TargetArray, params: Params
        ) -> Num[Array, "..."]:
            return jnp.sum((model_fn(input, params) - target) ** 2)

        return loss_wrapper

    if loss_fn == LossFn.CROSSENTROPY:

        def loss_wrapper(
            input: InputArray, target: TargetArray, params: Params
        ) -> Num[Array, "..."]:
            return jax.lax.log_sigmoid_cross_entropy(model_fn(input, params), target)

        return loss_wrapper

    if callable(loss_fn):

        def loss_wrapper(
            input: InputArray, target: TargetArray, params: Params
        ) -> Num[Array, "..."]:
            return loss_fn(model_fn(input, params), target)

        return loss_wrapper

    msg = f"Unknown loss function: {loss_fn}."
    raise ValueError(msg)


def create_hessian_mv_without_data(
    model_fn: ModelFn,  # type: ignore[reportRedeclaration]
    params: Params,
    loss_fn: LossFn | Callable | None = None,
    *,
    has_batch: bool = False,
    **kwargs,
) -> Callable[[Params, Data], Params]:
    r"""Computes the Hessian-vector product (HVP) for a model and loss function.

    This function computes the HVP by combining the model and loss functions into a
    single callable. It evaluates the Hessian at the provided model parameters, with
    respect to the model and loss function.

    Mathematically:
    $$H \cdot v = \nabla^2 L(x, y, \theta) \cdot v$$
    where $L$ is the combined loss function, $\theta$ are the parameters, and $v$ is the
    input vector.

    Args:
        model_fn: The model function to evaluate.
        params: The parameters of the model.
        loss_fn: The loss function to apply. Supported options are:
            - `LossFn.MSE` for mean squared error.
            - `LossFn.CROSSENTROPY` for cross-entropy loss.
            - A custom callable loss function.
        has_batch: Whether the model function should be vectorized over the batch.
        **kwargs: Additional arguments (ignored).

    Returns:
        A function that computes the HVP for a given vector and batch of data.
    """
    del kwargs

    new_model_fn: Callable[[InputArray, TargetArray, Params], Num[Array, "..."]]  # noqa: UP037

    if loss_fn is not None:
        new_model_fn = concatenate_model_and_loss_fn(
            model_fn, loss_fn, has_batch=has_batch
        )
    else:

        def model_without_loss(
            input: InputArray, target: TargetArray, params: Params
        ) -> PredArray:
            del target  # Ignore target since there's no loss
            return model_fn(input, params)

        new_model_fn = model_without_loss

    def _hessian_mv(vector: Params, data: Data) -> Params:
        return hvp(
            lambda p: new_model_fn(data["input"], data["target"], p),
            params,
            vector,
        )

    return _hessian_mv


def create_hessian_mv(
    model_fn: ModelFn,  # type: ignore[reportRedeclaration]
    params: Params,
    data: Data,
    loss_fn: LossFn | Callable | None = None,
    **kwargs,
) -> Callable[[Params], Params]:
    r"""Computes the Hessian-vector product (HVP) for a model and loss fn. with data.

    This function wraps `create_hessian_mv_without_data`, fixing the dataset to produce
    a function that computes the HVP for the specified data.

    Mathematically:
    $$ H \cdot v = \nabla^2 L(x, y, \theta) \cdot v $$
    where $L$ is the combined loss function, $\theta$ are the parameters, and $v$ is the
    input vector of the HVP.

    Args:
        model_fn: The model function to evaluate.
        params: The parameters of the model.
        data: A batch of input and target data.
        loss_fn: The loss function to apply. Supported options are:
            - `LossFn.MSE` for mean squared error.
            - `LossFn.CROSSENTROPY` for cross-entropy loss.
            - A custom callable loss function.
        **kwargs: Additional arguments (ignored).

    Returns:
        A function that computes the HVP for a given vector and the fixed dataset.
    """
    hessian_mv = create_hessian_mv_without_data(model_fn, params, loss_fn, **kwargs)
    return lambda vector: hessian_mv(vector, data)
