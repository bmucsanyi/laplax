import jax
import jax.numpy as jnp
from flax import nnx

from laplax.curv.cov import create_full_cov
from laplax.curv.ggn import create_ggn_mv
from laplax.eval.push_forward import set_mc_pushforward
from laplax.util.flatten import create_partial_pytree_flattener
from laplax.util.mv import todense
from laplax.util.tree import allclose, get_size, ones_like
from flax import linen as nn
import equinox as eqx
from pytest_cases import parametrize_with_cases

# use for one of: "linen", "nnx" or "equinox
def case_linen():
    class MLP(nn.Module):
        def setup(self, in_channels=1, hidden_channels=20, out_channels=1):
            # Define the layers in the init method
            self.linear1 = nn.Dense(hidden_channels)
            self.linear2 = nn.Dense(out_channels)

        def __call__(self, x):
            # Apply the layers to the input x
            x = self.linear1(x)
            x = nn.tanh(x)
            x = self.linear2(x)
            return x

    key = jax.random.key(0)
    rngs = nnx.Rngs(0)
    data = {
        "input": jax.random.normal(key, (1, 10)),
        "target": jax.random.normal(key, (1, 1)),
    }

    model = MLP(in_channels=10, hidden_channels=20, out_channels=1)

    model_fn = lambda params, x: model.apply(params, input)
    graph_def, _ = nnx.split(model)
    return model, data, model_fn


def case_nnx():
    class MLP(nnx.Module):
        def __init__(self, rngs, in_channels=1, hidden_channels=20, out_channels=1):
            self.linear1 = nnx.Linear(in_channels, hidden_channels, rngs=rngs)
            self.tanh = nnx.tanh
            self.linear2 = nnx.Linear(hidden_channels, out_channels, rngs=rngs)

        def __call__(self, x):
            x = self.linear1(x)
            x = self.tanh(x)
            x = self.linear2(x)
            return x

    key = jax.random.key(0)
    rngs = nnx.Rngs(0)
    data = {
        "input": jax.random.normal(key, (1, 10)),
        "target": jax.random.normal(key, (1, 1)),
    }

    model = MLP(in_channels=10, hidden_channels=20, out_channels=1, rngs=rngs)
    graph_def, _ = nnx.split(model)

    def model_fn(params, input):
        return nnx.call((graph_def, params))(input)[0]

    return model, data, model_fn

"""
def case_equinox():
    class MLP(eqx.Module):
        layers: list

        def __init__(self, rngs, in_channels=1, hidden_channels=20, out_channels=1):
            self.layers = [eqx.nn.Linear(in_channels, hidden_channels, rngs),
                           eqx.nn.Linear(hidden_channels, out_channels, rngs]

        def __call__(self, x):
            for layer in self.layers[:-1]:
                x = jax.nn.tanh(layer(x))
            return self.layers[-1](x)

    key = jax.random.key(0)
    rngs = nnx.Rngs(0)
    data = {
        "input": jax.random.normal(key, (1, 10)),
        "target": jax.random.normal(key, (1, 1)),
    }

    model = MLP(in_channels=10, hidden_channels=20, out_channels=1, rngs=rngs)
    return model, data
"""


@parametrize_with_cases("model, data, model_fn", cases='.')
def test_push_forward(model, data, model_fn):
    graph_def, params, rest = nnx.split(model, nnx.Param, ...)


    # Calculate ggn
    ggn_mv = create_ggn_mv(model_fn, params, data, "mse")
    n_params = get_size(params)
    ggn = todense(ggn_mv, like=params)

    # Testing
    get_posterior = create_full_cov(ggn_mv, tree=params)
    cov = get_posterior(prior_prec=1.0, return_scale=True)

    # Set pushforward
    pushforward = set_mc_pushforward(
        key=jax.random.key(0),
        model_fn=model_fn,
        mean=params,
        posterior=get_posterior,
        prior_prec=99999999.0,
        n_weight_samples=100000,
    )

    # Compute pushforward
    results = pushforward(data["input"])

    assert jnp.allclose(model_fn(params, data["input"]), results["pred"])
    assert jnp.allclose(model_fn(params, data["input"]), results["pred_mean"], rtol=1e-2)


