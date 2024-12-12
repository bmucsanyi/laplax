from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import pytest_cases

from laplax import util
from laplax.curv.cov import (
    CURVATURE_METHODS,
    CURVATURE_PRIOR_METHODS,
    CURVATURE_STATE_TO_COV,
    CURVATURE_STATE_TO_SCALE,
    CURVATURE_TO_POSTERIOR_STATE,
    prec_to_scale,
    register_curvature_method,
)

from .cases.covariance import case_posterior_covariance


@pytest.fixture(
    params=[pytest.param(seed, id=f"seed{seed}") for seed in [1, 42, 256]],
)
def key(request) -> np.random.Generator:
    return jax.random.key(request.param)


@pytest_cases.fixture
def prior_prec():
    """Fixture for precision matrix."""
    return jnp.array([[3.0, 0.5], [0.5, 2.0]])


@pytest_cases.fixture
def invalid_prec():
    """Fixture for invalid precision matrix."""
    return jnp.array([[1.0, 2.0], [2.0, 1.0]])


# --------------------------------------------------------------------------------------
# Test cases
# --------------------------------------------------------------------------------------


def test_prec_to_scale(prior_prec):
    """Test `prec_to_scale` for valid input."""
    scale = prec_to_scale(prior_prec)
    assert (
        scale.shape == prior_prec.shape
    ), "Scale matrix shape should match precision matrix shape."
    assert jnp.all(
        jnp.linalg.eigvals(scale) > 0
    ), "Scale matrix should be positive definite."
    assert jnp.allclose(
        scale @ scale.T @ prior_prec, jnp.eye(prior_prec.shape[0]), atol=1e-6, rtol=1e-6
    )


def test_prec_to_scale_invalid(invalid_prec):
    """Test `prec_to_scale` for invalid input."""
    with pytest.raises(ValueError, match="Matrix is not positive definite"):
        prec_to_scale(invalid_prec)


@pytest_cases.parametrize_with_cases("task", cases=case_posterior_covariance)
def test_posterior_covariance_est(task):
    # Get low rank terms
    curv_est = CURVATURE_METHODS[task.method](
        mv=task.arr_mv,
        layout=task.tree_like,
        key=task.key_curv_est,
        maxiter=task.rank,
    )
    assert jnp.allclose(
        task.adjust_curv_est(curv_est),
        task.true_curv,
        atol=1e-2,
        rtol=1e-2,
    )

    # Get and test precision matrix
    prec = CURVATURE_PRIOR_METHODS[task.method](curv_est, prior_prec=1.0)
    prec_dense = task.adjust_prec(prec)
    assert jnp.allclose(
        prec_dense, task.true_curv + jnp.eye(task.size), atol=1e-4, rtol=1e-4
    )

    # Create posterior state
    state = CURVATURE_TO_POSTERIOR_STATE[task.method](prec)

    # Get and test scale matrix
    scale_mv = CURVATURE_STATE_TO_SCALE[task.method](state)
    scale_dense = util.mv.todense(scale_mv, layout=task.tree_like)
    assert jnp.allclose(
        scale_dense @ scale_dense.T @ prec_dense,
        jnp.eye(task.size),
        atol=1e-2,
        rtol=1e-2,
    )

    # Get and test covariance matrix
    cov_mv = CURVATURE_STATE_TO_COV[task.method](state)
    cov_dense = util.mv.todense(cov_mv, layout=task.tree_like)
    assert jnp.allclose(
        cov_dense @ prec_dense, jnp.eye(task.size), atol=1e-2, rtol=1e-2
    )


# -------------------------------------------------------------------------------
# Test register_curvature_method
# -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "create_fn", "prior_fn", "posterior_fn", "scale_fn", "cov_fn", "default"),
    [
        (
            "test_method",
            Mock(name="create_fn"),
            Mock(name="prior_fn"),
            Mock(name="posterior_fn"),
            Mock(name="scale_fn"),
            Mock(name="cov_fn"),
            None,
        ),
    ],
)
def test_register_curvature_method(  # noqa: PLR0913, PLR0917
    name, create_fn, prior_fn, posterior_fn, scale_fn, cov_fn, default
):
    register_curvature_method(
        name=name,
        create_fn=create_fn,
        prior_fn=prior_fn,
        posterior_fn=posterior_fn,
        scale_fn=scale_fn,
        cov_fn=cov_fn,
        default=default,
    )

    assert CURVATURE_METHODS[name] == create_fn
    assert CURVATURE_PRIOR_METHODS[name] == prior_fn
    assert CURVATURE_TO_POSTERIOR_STATE[name] == posterior_fn
    assert CURVATURE_STATE_TO_SCALE[name] == scale_fn
    assert CURVATURE_STATE_TO_COV[name] == cov_fn


@pytest.mark.parametrize(
    ("name", "default"),
    [
        ("default_test", "low_rank"),
    ],
)
def test_register_curvature_method_with_default(name, default):
    register_curvature_method(name=name, default=default)

    assert CURVATURE_METHODS[name] == CURVATURE_METHODS[default]
    assert CURVATURE_PRIOR_METHODS[name] == CURVATURE_PRIOR_METHODS[default]
    assert CURVATURE_TO_POSTERIOR_STATE[name] == CURVATURE_TO_POSTERIOR_STATE[default]
    assert CURVATURE_STATE_TO_SCALE[name] == CURVATURE_STATE_TO_SCALE[default]
    assert CURVATURE_STATE_TO_COV[name] == CURVATURE_STATE_TO_COV[default]


def test_register_curvature_method_missing_functions():
    with pytest.raises(ValueError, match="must be specified"):
        register_curvature_method(name="incomplete_test")


@pytest.mark.parametrize(
    ("name", "create_fn", "default"),
    [
        ("partial_test", Mock(name="create_fn"), "low_rank"),
    ],
)
def test_register_curvature_method_partial(name, create_fn, default):
    register_curvature_method(name=name, create_fn=create_fn, default=default)

    assert CURVATURE_METHODS[name] == create_fn
    assert CURVATURE_PRIOR_METHODS[name] == CURVATURE_PRIOR_METHODS[default]
    assert CURVATURE_TO_POSTERIOR_STATE[name] == CURVATURE_TO_POSTERIOR_STATE[default]
    assert CURVATURE_STATE_TO_SCALE[name] == CURVATURE_STATE_TO_SCALE[default]
    assert CURVATURE_STATE_TO_COV[name] == CURVATURE_STATE_TO_COV[default]
