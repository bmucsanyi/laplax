"""Pushforward functions for weight space uncertainty.

This file contains the additional functions for pushing forward
weight space uncertainty onto output uncertainty.
"""

import math

import jax
import jax.numpy as jnp

from laplax import util
from laplax.curv.cov import Posterior
from laplax.eval.utils import finalize_functions
from laplax.types import (
    Any,
    Array,
    Callable,
    DistState,
    InputArray,
    KeyType,
    ModelFn,
    Params,
    PosteriorState,
    PredArray,
    PriorArguments,
)
from laplax.util.ops import lmap, precompute_list

# -------------------------------------------------------------------------
# Utilities - General
# -------------------------------------------------------------------------


def set_get_weight_sample(key, mean, scale_mv, n_weight_samples, **kwargs):
    keys = jax.random.split(key, n_weight_samples)

    def get_weight_sample(idx):
        return util.tree.normal_like(keys[idx], mean, scale_mv)

    return precompute_list(
        get_weight_sample,
        jnp.arange(n_weight_samples),
        option=kwargs.get("precompute_samples", "samples"),
    )


# -------------------------------------------------------------------------
# Posterior state to distribution state
# -------------------------------------------------------------------------


def get_dist_state(
    mean: Params,
    model_fn: ModelFn,
    posterior_state: PosteriorState,
    *,
    linearized: bool = False,
    n_samples: int = 0,
    key: KeyType | None = None,
) -> DistState:
    dist_state = {
        "posterior_state": posterior_state,
        "n_samples": n_samples,
    }

    if linearized:
        # Create pushforward functions
        def pf_jvp(input: InputArray, vector: Params) -> PredArray:
            return jax.jvp(
                lambda p: model_fn(input=input, params=p),
                (mean,),
                (vector,),
            )[1]

        def pf_vjp(input: InputArray, vector: PredArray) -> Params:
            out, vjp_fun = jax.vjp(lambda p: model_fn(input=input, params=p), mean)
            return vjp_fun(vector.reshape(out.shape))

        dist_state["vjp"] = pf_vjp
        dist_state["jvp"] = pf_jvp

    if n_samples > 0:
        # Create weight sample function
        get_weight_samples = set_get_weight_sample(
            key,
            mean,
            posterior_state.scale_mv(posterior_state.state),
            n_samples,
        )
        dist_state["get_weight_samples"] = get_weight_samples

    return dist_state


# -------------------------------------------------------------------------
# Utilities - Ensemble pushforward
# -------------------------------------------------------------------------


def mc_setup(
    results: dict[str, Array],
    aux: dict[str, Any],
    input: InputArray,
    dist_state: DistState,
    **kwargs,
) -> tuple[dict[str, Array], dict[str, Any]]:
    def compute_pred_ptw(idx: int) -> PredArray:
        weight_sample = dist_state["get_weight_samples"](idx)
        return aux["model_fn"](input=input, params=weight_sample)

    aux["pred_ensemble"] = lmap(
        compute_pred_ptw,
        jnp.arange(dist_state["n_samples"]),
        batch_size=kwargs.get("lmap_pred_ptw", "weight"),
    )

    return results, aux


def mc_pred_mean(
    results: dict[str, Array], aux: dict[str, Any], name: str, **kwargs
) -> tuple[dict[str, Array], dict[str, Any]]:
    """Compute mean of ensemble predictions."""
    del kwargs
    pred_ensemble = aux.get("pred_ensemble")
    results[name] = util.tree.mean(pred_ensemble, axis=0)
    return results, aux


def mc_pred_cov(
    results: dict[str, Array], aux: dict[str, Any], name: str, **kwargs
) -> tuple[dict[str, Array], dict[str, Any]]:
    """Compute covariance of ensemble predictions."""
    del kwargs
    pred_ensemble = aux.get("pred_ensemble")
    if pred_ensemble is None:
        msg = "Prediction ensemble is not available."
        raise ValueError(msg)

    results[name] = util.tree.cov(
        pred_ensemble.reshape(pred_ensemble.shape[0], -1), rowvar=False
    )
    return results, aux


def mc_pred_var(
    results: dict[str, Array], aux: dict[str, Any], name: str, **kwargs
) -> tuple[dict[str, Array], dict[str, Any]]:
    """Compute variance of ensemble predictions."""
    del kwargs
    if "pred_cov" in results:
        results[name] = jnp.diagonal(results["pred_cov"])
    else:
        pred_ensemble = aux.get("pred_ensemble")
        results[name] = util.tree.var(pred_ensemble, axis=0)
    return results, aux


def mc_pred_std(
    results: dict[str, Array], aux: dict[str, Any], name: str, **kwargs
) -> tuple[dict[str, Array], dict[str, Any]]:
    """Compute standard deviation of ensemble predictions."""
    del kwargs
    if "pred_var" in results:
        results[name] = jnp.sqrt(results["pred_var"])
    else:
        pred_ensemble = aux.get("pred_ensemble")
        results[name] = util.tree.std(pred_ensemble, axis=0)
    return results, aux


def mc_samples(
    results: dict[str, Array],
    aux: dict[str, Any],
    name: str,
    n_samples: int = 5,
    **kwargs,
) -> tuple[dict[str, Array], dict[str, Any]]:
    """Select samples from ensemble."""
    del kwargs
    pred_ensemble = aux.get("pred_ensemble")
    results[name] = util.tree.tree_slice(pred_ensemble, 0, n_samples)
    return results, aux


DEFAULT_MC_FUNCTIONS = {
    "pred_ensemble": mc_setup,
    "pred_mean": mc_pred_mean,
    "pred_var": mc_pred_var,
    "pred_std": mc_pred_std,
    "pred_cov": mc_pred_cov,
    "samples": mc_samples,
}

# -------------------------------------------------------------------------
# Utilities - Linearized pushforward
# -------------------------------------------------------------------------


def set_output_cov_mv(
    posterior_state: Posterior,
    input: InputArray,
    jvp: Callable[[InputArray, Params], PredArray],
    vjp: Callable[[InputArray, PredArray], Params],
):
    cov_mv = posterior_state.cov_mv(posterior_state.state)
    scale_mv = posterior_state.scale_mv(posterior_state.state)

    def output_cov_mv(vec: PredArray) -> PredArray:
        return jvp(input, cov_mv(vjp(input, vec)[0]))

    def output_cov_scale_mv(vec: PredArray) -> PredArray:
        return jvp(input, scale_mv(vec))

    return {"cov_mv": output_cov_mv, "scale_mv": output_cov_scale_mv}


def lin_setup(
    results: dict[str, Array],
    aux: dict[str, Any],
    input: InputArray,
    dist_state: DistState,
    **kwargs,
) -> tuple[dict[str, Array], dict[str, Any]]:
    del kwargs

    jvp = dist_state.get("jvp")
    vjp = dist_state.get("vjp")
    posterior_state = dist_state.get("posterior_state")

    # Check types (mainly needed for type checker)
    if not isinstance(posterior_state, Posterior):
        msg = "Posterior state is not a Posterior type."
        raise TypeError(msg)

    if not isinstance(jvp, Callable):
        msg = "JVP is not a JVPType."
        raise TypeError(msg)

    if not isinstance(vjp, Callable):
        msg = "VJP is not a VJPType."
        raise TypeError(msg)

    mv = set_output_cov_mv(posterior_state, input, jvp, vjp)
    aux["cov_mv"] = mv.get("cov_mv")
    aux["scale_mv"] = mv.get("scale_mv")

    return results, aux


def lin_pred(
    results: dict[str, Array],
    aux: dict[str, Any],
    name: str,
    **kwargs,
) -> tuple[dict[str, Array], dict[str, Any]]:
    del kwargs
    pred = results.get("pred")

    if pred is None:
        msg = "Pred is not a PredArray."
        raise TypeError(msg)

    results[name] = pred
    return results, aux


def lin_pred_var(
    results: dict[str, Array],
    aux: dict[str, Any],
    name: str,
    **kwargs,
) -> tuple[dict[str, Array], dict[str, Any]]:
    del kwargs
    cov = results.get("pred_cov", aux.get("cov_mv"))
    pred = results.get("pred")

    # Check types (mainly needed for type checker)
    if cov is None:
        msg = "Covariance is not available."
        raise ValueError(msg)

    if pred is None:
        msg = "Prediction is not available."
        raise ValueError(msg)

    # Compute diagonal as variance
    results[name] = util.mv.diagonal(cov, layout=math.prod(pred.shape))
    return results, aux


def lin_pred_std(
    results: dict[str, Array],
    aux: dict[str, Any],
    name: str,
    **kwargs,
) -> tuple[dict[str, Array], dict[str, Any]]:
    del kwargs
    var = results.get("pred_var")
    results[name] = util.tree.sqrt(var)
    return results, aux


def lin_pred_cov(
    results: dict[str, Array],
    aux: dict[str, Any],
    name: str,
    **kwargs,
) -> tuple[dict[str, Array], dict[str, Any]]:
    del kwargs
    pred = results.get("pred")
    cov_mv = aux.get("cov_mv")

    # Check types (mainly needed for type checker)
    if not isinstance(cov_mv, Callable):
        msg = f"cov_mv (type: {type(cov_mv)}) is not of type Callable."
        raise TypeError(msg)

    results[name] = util.mv.todense(cov_mv, layout=pred)
    return results, aux


def lin_n_samples_fn(
    results: dict[str, Array],
    aux: dict[str, Any],
    dist_state: DistState,
    name: str,
    **kwargs,
):
    # Unpack arguments
    scale_mv = aux.get("scale_mv")
    get_weight_samples = dist_state.get("get_weight_samples")
    n_samples = dist_state.get("n_samples")

    # Check types (mainly needed for type checker)
    if not isinstance(scale_mv, Callable):
        msg = f"scale_mv (type: {type(scale_mv)}) is not of type Callable."
        raise TypeError(msg)

    if not isinstance(get_weight_samples, Callable):
        msg = (
            f"get_weight_samples (type: {type(get_weight_samples)}) "
            f"is not of type Callable."
        )
        raise TypeError(msg)

    if not isinstance(n_samples, int):
        msg = f"n_samples (type: {type(n_samples)}) is not of type int."
        raise TypeError(msg)

    # Compute samples
    results[name] = lmap(
        lambda i: scale_mv(get_weight_samples(i)),
        jnp.arange(n_samples),
        batch_size=kwargs.get("lmap_lin_samples", "weight"),
    )
    return results, aux


DEFAULT_LIN_FINALIZE = {
    "setup": lin_setup,
    "pred_mean": lin_pred,
    "pred_var": lin_pred_var,
    "pred_std": lin_pred_std,
    "pred_cov": lin_pred_cov,
    "samples": lin_n_samples_fn,
}


# -------------------------------------------------------------------------
# Pushforward functions
# -------------------------------------------------------------------------


def set_prob_predictive(
    model_fn: ModelFn,
    mean: Params,
    dist_state: DistState,
    pushforward_functions: dict,
    **kwargs,
) -> Callable[[InputArray], dict[str, Array]]:
    def prob_predictive(input: InputArray) -> dict[str, Array]:
        # Mean prediction
        pred = model_fn(input=input, params=mean)
        aux = {"model_fn": model_fn, "mean": mean}

        # Compute prediction
        return finalize_functions(
            functions=pushforward_functions,
            results={"pred": pred},
            dist_state=dist_state,
            aux=aux,
            input=input,
            **kwargs,
        )

    return prob_predictive


def set_mc_pushforward(
    model_fn: ModelFn,
    mean: Params,
    posterior: Callable[[PriorArguments], Posterior],
    prior_arguments: PriorArguments,
    *,
    key: KeyType,
    pushforward_functions: dict = DEFAULT_MC_FUNCTIONS,
    n_weight_samples: int = 100,
    **kwargs,
):
    # Create weight sample function
    posterior_state = posterior(prior_arguments)

    # Posterior state to dist_state
    dist_state = get_dist_state(
        mean,
        model_fn,
        posterior_state,
        linearized=False,
        n_samples=n_weight_samples,
        key=key,
    )

    # Set prob predictive
    prob_predictive = set_prob_predictive(
        model_fn=model_fn,
        mean=mean,
        dist_state=dist_state,
        pushforward_functions=pushforward_functions,
        **kwargs,
    )

    return prob_predictive


def set_lin_pushforward(
    model_fn: ModelFn,
    mean: Params,
    posterior: Callable[[PriorArguments], Posterior],
    prior_arguments: PriorArguments,
    pushforward_functions: dict = DEFAULT_LIN_FINALIZE,
    **kwargs,
) -> Callable:
    # Create posterior state
    posterior_state = posterior(prior_arguments)

    # Posterior state to dist_state
    dist_state = get_dist_state(
        mean,
        model_fn,
        posterior_state,
        linearized=True,
        n_samples=kwargs.get("n_samples", 0),
        key=kwargs.get("key"),
    )

    # Set prob predictive
    prob_predictive = set_prob_predictive(
        model_fn=model_fn,
        mean=mean,
        dist_state=dist_state,
        pushforward_functions=pushforward_functions,
        **kwargs,
    )

    return prob_predictive


def set_posterior_gp_kernel(
    model_fn: ModelFn,
    mean: Params,
    posterior: Callable[..., Posterior],
    prior_arguments: PriorArguments,
    **kwargs,
) -> tuple[Callable, DistState]:
    # Create posterior state
    posterior_state = posterior(prior_arguments)

    # Posterior state to dist_state
    dist_state = get_dist_state(
        mean,
        model_fn,
        posterior_state,
        linearized=True,
        n_samples=0,
    )

    # Kernel mv
    def kernel_mv(
        vec: PredArray, x1: InputArray, x2: InputArray, dist_state: dict[str, Any]
    ) -> PredArray:
        cov_mv = dist_state["posterior_state"].cov_mv(
            dist_state["posterior_state"].state
        )
        return dist_state["jvp"](x1, cov_mv(dist_state["vjp"](x2, vec)[0]))

    if kwargs.get("dense", False):
        output_layout = kwargs.get("output_layout")
        if output_layout:
            return lambda x1, x2: util.mv.todense(
                lambda v: kernel_mv(v, x1, x2, dist_state), layout=output_layout
            ), dist_state
        msg = (
            "Function should return a dense matrix, but no output layout is specified."
        )
        raise ValueError(msg)

    return kernel_mv, dist_state
