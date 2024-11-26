import jax
import jax.numpy as jnp
from flax import nnx

from typing import List, Callable, Any
from laplax.curv.ggn import create_ggn_mv
from laplax.eval.push_forward import set_mc_pushforward
from laplax.util.flatten import create_partial_pytree_flattener
from laplax.util.mv import todense
from laplax.util.tree import allclose, get_size, ones_like
from functools import partial

from flax import linen as nn
import equinox as eqx
from pytest_cases import parametrize_with_cases
from jax import random
from .cases.classification import case_classification

"""
class Regression:
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
                for layer in self.layers[:-1]:
                    x = layer(x)
                    x = self.activation(x)
                return self.final_activation(self.layers[-1](x))

        key = jax.random.key(2)
        data = {
            "input": jax.random.normal(key, (1, 10)),
            "target": jax.random.normal(key, (1, 1)),
        }
        keys = [jax.random.key(0), jax.random.key(1)]
        model = MLP(keys=keys, in_channels=10, hidden_channels=20, out_channels=1)

        def model_fn(params, input):
            if input.ndim == 1:
                input = jnp.expand_dims(input, axis=0)
                # Enforce batch size and input dimensions
            new_model = eqx.tree_at(lambda m: m, model, params)
            return jax.vmap(new_model)(input)

        params = eqx.filter(model, eqx.is_inexact_array)
        return model_fn, data, params

class Classification:
# use for one of: "linen", "nnx" or "equinox
    def case_linen(self):
        class CNN(nn.Module):
            def setup(self):
                self.conv1 = nn.Conv(features=2, kernel_size=(3, 3))  # Conv2D layer
                self.linear1 = nn.Dense(features=10)  # Dense (fully connected) layer

            def __call__(self, x):
                # Ensure x has 4 dimensions (batch_size, height, width, channels)
                if x.ndim == 3:
                    x = jnp.expand_dims(x, axis=0)

                # Apply convolution, ReLU, and average pooling
                x = nn.relu(self.conv1(x))
                x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

                # Flatten spatial dimensions
                x = x.reshape((x.shape[0], -1))  # Shape: (batch_size, flattened_features)

                # Apply dense layer and ReLU
                x = nn.relu(self.linear1(x))

                return x

        key = jax.random.key(0)
        rngs = nnx.Rngs(0)
        data = {
            # Single grayscale image with shape (1, 8, 8, 1) for batch size 1
            "input": jax.random.normal(key, (1, 8, 8, 1)),

            # Random one-hot target with 10 classes
            "target": jax.nn.one_hot(jax.random.randint(key, (1,), 0, 10), num_classes=10),
        }

        model = CNN()
        rng_key = random.key(711)
        parameters = model.init(rng_key, random.normal(key, (1, 8, 8, 1)))

        def model_fn(params, input):
            return model.apply(params, input)

        params = parameters

        return model_fn, data, params

    def case_nnx(self):
        class CNN(nnx.Module):


            def __init__(self, rngs: nnx.Rngs):
                self.conv1 = nnx.Conv(1, 2, kernel_size=(3, 3), rngs=rngs)
                self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
                self.linear1 = nnx.Linear(32, 10, rngs=rngs)

            def __call__(self, x):
                if x.ndim == 3:
                    x = jnp.expand_dims(x, axis=0)
                x = self.avg_pool(nnx.relu(self.conv1(x)))
                x = x.reshape(x.shape[0], -1)  # flatten
                x = nnx.relu(self.linear1(x))
                return x

        key = jax.random.key(0)
        rngs = nnx.Rngs(0)
        data = {
            # Single grayscale image with shape (1, 8, 8, 1) for batch size 1
            "input": jax.random.normal(key, (1, 8, 8, 1)),

            # Random one-hot target with 10 classes
            "target": jax.nn.one_hot(jax.random.randint(key, (1,), 0, 10), num_classes=10),
        }

        model = CNN(rngs=rngs)

        def model_fn(params, input):
            nnx.update(model, nnx.GraphState.merge(params, rest))  # Load the model from parameters
            return model(input)

        _, params, rest = nnx.split(model, nnx.Param, ...)
        return model_fn, data, params

    def case_equinox(self):

        class CNN(eqx.Module):
            layers: list
            activation: Callable = eqx.static_field()
            final_activation: Callable = eqx.static_field()

            def __init__(self, key):
                key1, key2 = jax.random.split(key, 2)
                # Standard CNN setup: convolutional layer, followed by flattening,
                # with a small MLP on top.
                self.layers = [
                    eqx.nn.Conv2d(1, 2, kernel_size=3, key=key1),
                    eqx.nn.AvgPool2d(kernel_size=2, stride=2),
                    eqx.nn.Linear(32, 10, key=key2),
                ]
                self.activation = nn.relu
                self.final_activation = nn.relu

            def __call__(self, x):
                for layer in self.layers:
                    x = layer(x)
                return self.final_activation(x)

        key = jax.random.key(0)
        rngs = nnx.Rngs(0)
        data = {
            # Single grayscale image with shape (1, 8, 8, 1) for batch size 1
            "input": jax.random.normal(key, (1, 1, 1, 8, 8)),

            # Random one-hot target with 10 classes
            "target": jax.nn.one_hot(jax.random.randint(key, (1,), 0, 10), num_classes=10),
        }

        model = CNN(key=key)

        def model_fn(params, input):
            if input.ndim == 1:
                input = jnp.expand_dims(input, axis=0)
                # Enforce batch size and input dimensions
            new_model = eqx.tree_at(lambda m: m, model, params)
            return jax.vmap(new_model)(input)

        params = eqx.filter(model, eqx.is_inexact_array)
        return model_fn, data, params
"""

import jax
import jax.numpy as jnp
import pytest_cases

from laplax.curv.cov import create_posterior_function
from laplax.curv.ggn import create_ggn_mv
from laplax.eval.push_forward import set_lin_pushforward, set_mc_pushforward

from .cases.regression import case_regression


import jax
import jax.numpy as jnp
import pytest_cases

from laplax.curv.cov import create_posterior_function
from laplax.curv.ggn import create_ggn_mv
from laplax.eval.push_forward import set_lin_pushforward, set_mc_pushforward

from .cases.regression import case_regression

#TODO: readd full
@pytest_cases.parametrize(
    "curv_op",
    ["diagonal", "low_rank"],
)
@pytest_cases.parametrize_with_cases("task", cases=case_classification)
def test_mc_push_forward(curv_op, task):
    model_fn = task.get_model_fn()
    params = task.get_parameters()
    data = task.get_data_batch(batch_size=20)

    # Set get posterior function
    ggn_mv = create_ggn_mv(model_fn, params, data, task.loss_fn_type)
    get_posterior = create_posterior_function(
        curv_op,
        mv=ggn_mv,
        tree=params,
        key=jax.random.key(20),
        maxiter=20,
    )

    # Set pushforward
    pushforward = set_mc_pushforward(
        key=jax.random.key(0),
        model_fn=model_fn,
        mean=params,
        posterior=get_posterior,
        prior_prec=9999999.0,
        n_weight_samples=100000,
    )

    # Compute pushforwards
    # pushforward = jax.jit(pushforward)
    results = jax.vmap(pushforward)(data["input"])

    # # Check results
    pred = jax.vmap(lambda x: model_fn(params, x))(data["input"])
    assert (5, task.out_channels) == results["samples"].shape[1:]  # Check shape
    assert jnp.all(results["pred_std"] > 0) #TODO: > or >=? (sometimes a class is always 0 for the classification)
    assert jnp.allclose(pred, results["pred"])


@pytest_cases.parametrize(
    "curv_op",
    ["diagonal", "low_rank"],
)

@pytest_cases.parametrize_with_cases("task", cases=case_classification)
def test_lin_push_forward(curv_op, task):
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

    # Compute pushforward
    pushforward = jax.jit(pushforward)
    results = jax.vmap(pushforward)(data["input"])

    # Check results
    pred = jax.vmap(lambda x: model_fn(params, x))(data["input"])
    assert (5, task.out_channels) == results["samples"].shape[
        1:
    ]  # (batch, samples, out)
    jnp.allclose(pred, results["pred"])
    jnp.allclose(pred, results["pred_mean"], rtol=1e-2)


