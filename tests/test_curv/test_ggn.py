from functools import partial

import jax
import jax.numpy as jnp
import optax
import pytest
import pytest_cases

from laplax.curv.ggn import create_ggn_mv, create_loss_hessian_mv
from laplax.types import LossFn
from laplax.util.ops import lmap

from .cases.rosenbrock import RosenbrockCase

# ---------------------------------------------------------------
# Loss Hessian
# ---------------------------------------------------------------


def test_cross_entropy_loss_hessian():
    key = jax.random.key(0)
    target = jnp.asarray(0)
    logits = jax.random.normal(key, (10,))

    # Set loss hessian via autodiff
    hess_autodiff = jax.hessian(
        optax.softmax_cross_entropy_with_integer_labels,
    )(logits, target)

    # Set loss hessian via laplax mv
    hess_mv = create_loss_hessian_mv("cross_entropy")
    hess_laplax = jax.vmap(partial(hess_mv, pred=logits))(jnp.eye(10))

    assert jnp.allclose(hess_autodiff, hess_laplax, atol=1e-8)


def test_mse_loss_hessian():
    key = jax.random.key(0)
    keys = jax.random.split(key, 2)
    pred = jax.random.normal(keys[0], (10,))
    target = jax.random.normal(keys[1], (10,))

    # Set loss hessian via autodiff
    hess_autodiff = jax.hessian(
        lambda pred, target: jnp.sum((pred - target) ** 2),
    )(pred, target)

    # Set loss hessian via laplax mv
    #    hess_mv = create_loss_hessian_mv("mse")
    hess_mv = create_loss_hessian_mv(LossFn.MSE)
    hess_laplax = jax.vmap(partial(hess_mv, pred=pred))(jnp.eye(10))

    assert jnp.allclose(hess_autodiff, hess_laplax, atol=1e-8)


def test_callable_loss_hessian():
    key = jax.random.key(0)
    keys = jax.random.split(key, 3)
    pred = jax.random.normal(keys[0], (10,))
    target = jax.random.normal(keys[1], (10,))

    # Set random loss function
    random_arr = jax.random.normal(keys[2], (10,))

    def loss_func(pred, target):
        return jnp.sum(random_arr @ (pred - target) ** 3)

    # Set loss hessian via autodiff
    hess_autodiff = jax.hessian(loss_func)(pred, target)

    # Set loss hessian via laplax mv
    hess_mv = create_loss_hessian_mv(loss_func)
    hess_laplax = jax.vmap(partial(hess_mv, pred=pred, target=target))(jnp.eye(10))

    assert jnp.allclose(hess_autodiff, hess_laplax, atol=1e-8)


# ---------------------------------------------------------------
# GGN - Rosenbrock
# ---------------------------------------------------------------


@pytest.mark.parametrize("alpha", [1.0, 100.0])
@pytest.mark.parametrize("x", [jnp.array([1.0, 1.0]), jnp.array([2.5, 0.8])])
def case_rosenbrock(x, alpha):
    return RosenbrockCase(x, alpha)


@pytest_cases.parametrize_with_cases("rosenbrock", cases=[case_rosenbrock])
def test_ggn_rosenbrock(rosenbrock):
    # Setup ggn_mv
    ggn_mv = create_ggn_mv(
        model_fn=rosenbrock.model_fn,
        params=rosenbrock.x,
        data={"input": jnp.zeros(1), "target": jnp.zeros(1)},
        loss_fn=rosenbrock.loss_fn,
    )

    # Compute the GGN
    ggn_calc = lmap(ggn_mv, jnp.eye(2))

    # Compare with the manual GGN
    ggn_manual = rosenbrock.ggn_manual
    assert jnp.allclose(ggn_calc, ggn_manual)
