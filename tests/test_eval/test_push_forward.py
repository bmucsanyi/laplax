import jax
import jax.numpy as jnp
from flax import nnx

from laplax.curv.cov import create_full_cov
from laplax.curv.ggn import create_ggn_mv
from laplax.eval.push_forward import set_mc_pushforward
from laplax.util.flatten import create_partial_pytree_flattener
from laplax.util.mv import todense
from laplax.util.tree import allclose, get_size, ones_like


def load_simple_model():
    class Model(nnx.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, rngs):
            self.in_channels = in_channels
            self.hidden_channels = hidden_channels
            self.out_channels = out_channels

            self.linear1 = nnx.Linear(in_channels, hidden_channels, rngs=rngs)
            self.linear2 = nnx.Linear(hidden_channels, hidden_channels, rngs=rngs)
            self.linear3 = nnx.Linear(hidden_channels, out_channels, rngs=rngs)

        def __call__(self, x):
            x = self.linear1(x)
            x = nnx.relu(x)
            x = self.linear2(x)
            x = nnx.relu(x)
            x = self.linear3(x)
            return x

    key = jax.random.key(0)
    rngs = nnx.Rngs(0)
    model = Model(in_channels=10, hidden_channels=20, out_channels=1, rngs=rngs)
    data = {
        "input": jax.random.normal(key, (1, 10)),
        "target": jax.random.normal(key, (1, 1)),
    }
    return model, data


def test_push_forward():
    # Create simple model
    model, data = load_simple_model()
    graph_def, params = nnx.split(model)

    def model_fn(params, input):
        return nnx.call((graph_def, params))(input)[0]

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

    jnp.allclose(model_fn(params, data["input"]), results["pred"])
    jnp.allclose(model_fn(params, data["input"]), results["pred_mean"], rtol=1e-2)


test_push_forward()
