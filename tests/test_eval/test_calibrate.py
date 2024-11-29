from functools import partial

import jax
import jax.numpy as jnp
import pytest_cases

from laplax.curv.cov import create_posterior_function
from laplax.curv.ggn import create_ggn_mv
from laplax.eval.calibrate import (
    calibration_metric,
    evaluate_for_given_prior_prec,
    optimize_prior_prec,
)
from laplax.eval.push_forward import set_lin_pushforward

from .cases.regression import case_regression


@pytest_cases.parametrize("curv_op", ["full"])
@pytest_cases.parametrize_with_cases("task", cases=case_regression)
def test_lin_push_forward(curv_op, task):
    """Test for pipeline integration of calibration function."""
    model_fn = task.get_model_fn()
    params = task.get_parameters()
    data = task.get_data_batch(batch_size=20)

    # Set get posterior function
    ggn_mv = create_ggn_mv(model_fn, params, data, task.loss_fn_type)
    get_posterior = create_posterior_function(
        curv_op,
        ggn_mv,
        layout=params,
        key=jax.random.key(20),
        maxiter=20,
    )

    # Set pushforward for calibration objective
    set_prob_predictive = partial(
        set_lin_pushforward,
        key=jax.random.key(0),
        model_fn=model_fn,
        mean=params,
        posterior=get_posterior,
        n_samples=5,  # TODO(2bys): Find a better way of setting this.
    )

    def calibration_objective(prior_prec, data):
        return evaluate_for_given_prior_prec(
            prior_prec=jnp.asarray(prior_prec),
            data=data,
            set_prob_predictive=set_prob_predictive,
            metric=calibration_metric,
        )

    # Optimize
    prior_prec = optimize_prior_prec(
        objective=calibration_objective,
        data=data,
        grid_size=10,
    )

    # Calculate values for comparison.
    prior_prec_interval = jnp.logspace(
        start=-5.0,  # Default values
        stop=6.0,
        num=10,
    )

    # Calculate
    true_val = calibration_objective(prior_prec=prior_prec, data=data)
    comparison_prec = calibration_objective(
        prior_prec=prior_prec_interval[-1], data=data
    )
    assert true_val <= comparison_prec
