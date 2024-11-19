import jax
import pytest_cases

from laplax.curv.cov import create_posterior_function
from laplax.curv.ggn import create_ggn_mv
from laplax.eval.metrics import DEFAULT_REGRESSION_METRICS
from laplax.eval.push_forward import set_lin_pushforward
from laplax.eval.utils import evaluate_metrics_on_dataset

from .cases.regression import case_regression


@pytest_cases.parametrize(
    "curv_op",
    ["full"],
)
@pytest_cases.parametrize_with_cases("task", cases=case_regression)
def test_eval_metrics(curv_op, task):
    model_fn = task.get_model_fn()
    params = task.get_parameters()
    data = task.get_data_batch(batch_size=20)

    # Set get posterior function
    ggn_mv = create_ggn_mv(model_fn, params, data, task.loss_fn_type)
    get_posterior = create_posterior_function(
        curv_op,
        ggn_mv,
        tree=params,
        key=jax.random.key(20),
        maxiter=20,
    )

    # Set pushforward
    pushforward = set_lin_pushforward(
        key=jax.random.key(0),
        model_fn=model_fn,
        mean=params,
        posterior=get_posterior,
        prior_prec=99999999.0,
        n_samples=5,  # TODO(2bys): Find a better way of setting this.
    )

    results = jax.vmap(pushforward)(data["input"])
    results = evaluate_metrics_on_dataset(
        pushforward,
        data,
        metrics=DEFAULT_REGRESSION_METRICS,
    )

    assert all(results["rmse"] > 0)
    assert all(results["q"] > 0)
    comparison = next(iter(results.values())).shape
    assert all(k.shape == comparison for k in results.values())
