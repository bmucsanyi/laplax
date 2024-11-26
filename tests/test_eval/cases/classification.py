from typing import Any, Callable, List, Tuple

import equinox as eqx
import jax
import pytest_cases
from flax import linen as nn
from flax import nnx
from jax import numpy as jnp
from functools import partial


def generate_data(key, input_shape, target_shape):
    return {
        "input": jax.random.normal(key, input_shape),
        "target": jax.nn.one_hot(jax.random.randint(key, (target_shape[0],), 0, maxval=target_shape[1]),
                                 num_classes=target_shape[1]),
    }


class BaseClassificationTask:
    def __init__(
            self,
            in_channels: tuple,
            conv_features: int,
            conv_kernel_size: int,
            avg_pool_shape: int,
            avg_pool_strides: int,
            linear_in: int,
            out_channels: int,
            seed: int,
            framework: str,
    ):
        self.in_channels = in_channels
        self.conv_features = conv_features
        self.conv_kernel_size = conv_kernel_size
        self.avg_pool_shape = avg_pool_shape
        self.avg_pool_strides = avg_pool_strides
        self.linear_in = linear_in
        self.out_channels = out_channels
        self.seed = seed
        self.framework = framework
        self.loss_fn_type = "mse"
        self._initialize()

    def _initialize(self):
        msg = "This method must be implemented by subclasses."
        raise NotImplementedError(msg)

    def get_model_fn(self) -> Callable:
        msg = "This method must be implemented by subclasses."
        raise NotImplementedError(msg)

    def get_parameters(self) -> Any:
        return self.params

    def get_data_batch(self, batch_size: int) -> dict:
        key = jax.random.key(self.seed)
        return generate_data(
            key, (batch_size,) + self.in_channels, (batch_size, self.out_channels)
        )


class LinenClassificationTask(BaseClassificationTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, framework="linen")

    def _initialize(self):
        class CNN(nn.Module):
            in_channels: tuple
            conv_features: int
            conv_kernel_size: int
            avg_pool_shape: int
            avg_pool_strides: int
            out_channels: int

            def setup(self):
                self.conv1 = nn.Conv(features=self.conv_features,
                                     kernel_size=(self.conv_kernel_size, self.conv_kernel_size))
                self.linear1 = nn.Dense(features=self.out_channels)

            def __call__(self, x):
                # Ensure x has 4 dimensions (batch_size, height, width, channels)
                batch_dim = True
                if x.ndim == 3:
                    x = jnp.expand_dims(x, axis=0)
                    batch_dim = False

                x = nn.relu(self.conv1(x))
                x = nn.avg_pool(x, window_shape=(self.avg_pool_shape,self.avg_pool_shape), strides=(self.avg_pool_strides,self.avg_pool_strides))

                x = x.reshape((x.shape[0], -1))  # Shape: (batch_size, flattened_features)

                x = nn.relu(self.linear1(x))

                if not batch_dim:
                    return jnp.squeeze(x)
                else:
                    return x

        rng_key = jax.random.PRNGKey(self.seed)
        data = generate_data(rng_key, (1,) + self.in_channels, (1, self.out_channels))
        self.model = CNN(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            conv_features=self.conv_features,
            conv_kernel_size=self.conv_kernel_size,
            avg_pool_shape=self.avg_pool_shape,
            avg_pool_strides=self.avg_pool_strides,
        )
        self.params = self.model.init(rng_key, data["input"])

    def get_model_fn(self):
        def model_fn(params, input):
            return self.model.apply(params, input)

        return model_fn


class NNXClassificationTask(BaseClassificationTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, framework="nnx")

    def _initialize(self):
        class CNN(nnx.Module):

            def __init__(
                    self,
                    rngs:nnx.Rngs,
                    in_channels: tuple,
                    conv_features: int,
                    conv_kernel_size: int,
                    avg_pool_shape: int,
                    avg_pool_strides: int,
                    linear_in: int,
                    out_channels: int
            ):
                self.conv1 = nnx.Conv(in_channels[2], conv_features, kernel_size=(conv_kernel_size, conv_kernel_size), rngs=rngs)
                self.avg_pool = partial(nnx.avg_pool, window_shape=(avg_pool_shape, avg_pool_shape), strides=(avg_pool_strides, avg_pool_strides))
                self.linear1 = nnx.Linear(linear_in, out_channels, rngs=rngs)

            def __call__(self, x):
                batch_dim = True
                if x.ndim == 3:
                    x = jnp.expand_dims(x, axis=0)
                    batch_dim = False

                x = self.avg_pool(nnx.relu(self.conv1(x)))
                x = x.reshape(x.shape[0], -1)  # flatten
                x = nnx.relu(self.linear1(x))

                if not batch_dim:
                    return jnp.squeeze(x)
                else:
                    return x

        rngs = nnx.Rngs(self.seed)

        self.model = CNN(
            rngs=rngs,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            conv_features=self.conv_features,
            conv_kernel_size=self.conv_kernel_size,
            avg_pool_shape=self.avg_pool_shape,
            avg_pool_strides=self.avg_pool_strides,
            linear_in=self.linear_in,
        )
        _, self.params, _ = nnx.split(self.model, nnx.Param, ...)

    def get_model_fn(self):
        def model_fn(params, input):
            _, _, rest = nnx.split(self.model, nnx.Param, ...)
            nnx.update(self.model, nnx.GraphState.merge(params, rest))  # Load the model from parameters
            return self.model(input)

        return model_fn


def create_task(
        task_class,
        in_channels,
        conv_features,
        conv_kernel_size,
        avg_pool_shape,
        avg_pool_strides,
        out_channels,
        linear_in,
        seed
):
    """Factory function to create regression tasks."""
    return task_class(
        in_channels=in_channels,
        conv_features=conv_features,
        conv_kernel_size=conv_kernel_size,
        avg_pool_shape=avg_pool_shape,
        avg_pool_strides=avg_pool_strides,
        linear_in=linear_in,
        out_channels=out_channels,
        seed=seed,
    )


@pytest_cases.parametrize(
    "task_class", [NNXClassificationTask]
)
@pytest_cases.parametrize("in_channels", [(8, 8, 1)])
@pytest_cases.parametrize("conv_features", [2])
@pytest_cases.parametrize("conv_kernel_size", [3])
@pytest_cases.parametrize("avg_pool_shape", [2])
@pytest_cases.parametrize("avg_pool_strides", [2])
@pytest_cases.parametrize("linear_in", [32])
@pytest_cases.parametrize("out_channels", [10])
def case_classification(
        task_class,
        in_channels: tuple,
        conv_features: int,
        conv_kernel_size: int,
        avg_pool_shape: int,
        avg_pool_strides: int,
        linear_in: int,
        out_channels: int
):
    """Test regression tasks with multiple frameworks and parameter combinations."""
    seed = 42
    return task_class(in_channels, conv_features, conv_kernel_size, avg_pool_shape, avg_pool_strides, linear_in,
                      out_channels,
                      seed)
