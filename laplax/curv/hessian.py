"""Full hessian estimation."""

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
    return jax.jvp(jax.grad(func), (primals,), (tangents,))[1]


def concatenate_model_and_loss_fn(
    model_fn: ModelFn,  # type: ignore[reportRedeclaration]
    loss_fn: LossFn | Callable | None = None,
    *,
    has_batch: bool = False,
) -> Callable[[InputArray, TargetArray, Params], Num[Array, "..."]]:
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
    hessian_mv = create_hessian_mv_without_data(model_fn, params, loss_fn, **kwargs)
    return lambda vector: hessian_mv(vector, data)
