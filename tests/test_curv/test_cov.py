import jax
import jax.numpy as jnp
import numpy as np
import pytest
import pytest_cases

from laplax.curv.cov import (
    # create_diagonal_cov,
    # create_full_cov,
    # create_low_rank_cov,
    prec_to_scale,
)

# from laplax.util.mv import array_to_mv


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

# prec_to_scale()


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
