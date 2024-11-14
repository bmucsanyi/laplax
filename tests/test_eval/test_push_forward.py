import jax
import jax.numpy as jnp
from flax import nnx

from typing import List, Callable
from laplax.curv.cov import create_full_cov
from laplax.curv.ggn import create_ggn_mv
from laplax.eval.push_forward import set_mc_pushforward
from laplax.util.flatten import create_partial_pytree_flattener
from laplax.util.mv import todense
from laplax.util.tree import allclose, get_size, ones_like
from flax import linen as nn
import equinox as eqx
from pytest_cases import parametrize_with_cases
from jax import random

#TODO: als class, klasse case function mit inchannels etc. paranmetrisieren möglich

class Regression():
# use for one of: "linen", "nnx" or "equinox
    def case_linen(self):
        class MLP(nn.Module):
            hidden_channels: int = 20
            out_channels: int = 1

            def setup(self):
                # Define the layers in setup
                self.linear1 = nn.Dense(self.hidden_channels)
                self.linear2 = nn.Dense(self.out_channels)

            def __call__(self, x):
                # Apply the layers to the input x
                x = self.linear1(x)
                x = nn.tanh(x)
                x = self.linear2(x)
                return x

        rng_key = random.key(711)
        data = {
            "input": jax.random.normal(rng_key, (1, 10)),
            "target": jax.random.normal(rng_key, (1, 1)),
        }

        model = MLP()
        parameters = model.init(rng_key, jnp.ones([1, 10]))
        def model_fn(params, input):
            return model.apply(params, input)

        params = parameters


        return model_fn, data, params

    def case_nnx(self):
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

        def model_fn(params, input):
            nnx.update(model, nnx.GraphState.merge(params, rest))  # Load the model from parameters
            return model(input)

        _, params, rest = nnx.split(model, nnx.Param, ...)
        return model_fn, data, params

    def case_equinox(self):
        class MLP(eqx.Module):
            layers: List[eqx.nn.Linear]
            activation: Callable = eqx.static_field()
            final_activation: Callable = eqx.static_field()

            def __init__(self, in_channels, hidden_channels, out_channels, keys):
                self.layers = [
                    eqx.nn.Linear(in_channels, hidden_channels, key=keys[0]),
                    eqx.nn.Linear(hidden_channels, out_channels, key=keys[1])
                ]
                self.activation = jax.nn.tanh  # Example activation function
                self.final_activation = lambda x: x  # Identity for simplicity

            def __call__(self, x):
                # Apply layers sequentially with activation
                for layer in self.layers[:-1]:
                    x = self.activation(layer(x))
                return self.final_activation(self.layers[-1](x))

        key = jax.random.key(2)
        data = {
            "input": jax.random.normal(key, (1, 10)),
            "target": jax.random.normal(key, (1, 1)),
        }
        keys = [jax.random.key(0), jax.random.key(1)]
        model = MLP(keys=keys, in_channels=10, hidden_channels=20, out_channels=1)

        def model_fn(params, input): # Enforce batch size and input dimensions
            new_model = eqx.tree_at(lambda m: m, model, params)
            return new_model(input)

        params = eqx.filter(model, eqx.is_inexact_array)
        return model_fn, data, params


"""
class Classification():
# use for one of: "linen", "nnx" or "equinox
    def case_linen(self):
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

    def case_nnx(self):
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
        graph_def, parameters = nnx.split(model)

        def model_fn(params, input):
            return nnx.call((graph_def, params))(input)[0]

        return model, data, model_fn, parameters

    def case_equinox(self):
        class MLP(eqx.Module):
            layers: list

            def __init__(self, rngs, in_channels=1, hidden_channels=20, out_channels=1):
                self.layers = [eqx.nn.Linear(in_channels, hidden_channels, rngs),
                               eqx.nn.Linear(hidden_channels, out_channels, rngs)]

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
@parametrize_with_cases("model_fn, data, params", cases=[Regression])
def test_push_forward(model_fn, data, params):
    # Calculate ggn
    ggn_mv = create_ggn_mv(model_fn, params, data, "mse")
    #n_params = get_size(params)
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


