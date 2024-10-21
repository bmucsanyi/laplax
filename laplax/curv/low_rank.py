# noqa: D100
from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.sparse.linalg import lobpcg_standard

import laplax
from laplax.curv.ggn import GGN


def scalar_plus_low_rank_mv(x, scalar, U, S):
    return scalar * x + U @ (S[:, None] * (U.T @ x))


def scalar_plus_low_rank_invert(U, S, scalar):
    return partial(
        scalar_plus_low_rank_mv,
        scalar=1 / scalar,
        U=U,
        S=-S / (scalar * (S + scalar)),
    )


def set_prob_predictive_with_low_rank(
    model_fn, prior_scale, params_flat, inflate_params, low_rank_terms, **kwargs
):
    rng = jax.random.PRNGKey(0)
    n_weight_samples = 100
    n_params = low_rank_terms["U"].shape[0]
    params_true = deepcopy(params_flat)
    low_rank_mv = scalar_plus_low_rank_invert(
        U=low_rank_terms["U"], S=low_rank_terms["S"], scalar=prior_scale
    )
    weight_samples = (
        params_flat[None]
        + low_rank_mv(jax.random.normal(rng, (n_params, n_weight_samples))).T
    )
    save_ensemble = kwargs.get("mode", "metric") == "ensemble"

    def get_prob_predictive(input: jax.Array):
        pred = model_fn(params=inflate_params(params_true), input=input)
        pred_ensemble = jnp.asarray([
            model_fn(params=inflate_params(weight), input=input)
            for weight in weight_samples
        ])
        return {
            "pred": pred,
            "pred_mean": jax.numpy.mean(pred_ensemble, axis=0),
            "pred_std": jax.numpy.std(pred_ensemble, axis=0),
            "ensemble": pred_ensemble if save_ensemble else None,
        }

    return get_prob_predictive


def get_low_rank_approx_with_small_eigenvalues(ggn: GGN, maxiter: int = 200):
    b = laplax.curv.lanczos.lanczos_random_init(ggn.shape[:1])
    # b = reparamax.curv.lanczos.lanczos_averaged_vjp_init(ggn)
    with jax.experimental.enable_x64():
        D = laplax.curv.lanczos.lanczos_isqrt_full_reortho(
            ggn, b, maxiter=maxiter
        )  # maxiter is the target rank
        svd_result = jnp.linalg.svd(D, full_matrices=False)
        ggn_eigen = {
            "U": jnp.asarray(svd_result.U, dtype=b.dtype),
            "S": jnp.asarray(svd_result.S**-2, dtype=b.dtype),
        }
    # ggn.astype(jnp.float32)  # Params need to be jnp.float32 afterwards.
    # TODO(2bys): Find a general solution for dtype. Callables will not have types.
    return ggn_eigen


def get_low_rank_approx_with_large_eigenvalues(ggn: GGN, maxiter: int = 200):
    if ggn.shape[0] < maxiter * 5:
        # necessary assertion for lobpcg_standard function.
        maxiter = ggn.shape[0] // 5 - 1

    jax.config.update("jax_enable_x64", True)  # noqa: FBT003
    b = jax.random.normal(
        jax.random.key(2759847),
        (ggn.shape[0], maxiter),
        dtype=jnp.float64,
        # dtype=ggn.dtype,
        # TODO(2bys): Check whether we should also run this in jnp.float64
    )
    eigen_sketch, eigen_vec_sketch, _ = lobpcg_standard(ggn, b, m=maxiter)
    ggn_eigen = {
        "U": jnp.asarray(
            eigen_vec_sketch, dtype=jnp.float32
        ),  # Change to laplax_dtype()
        "S": jnp.asarray(eigen_sketch, dtype=jnp.float32),
    }
    jax.config.update("jax_enable_x64", False)  # noqa: FBT003
    return ggn_eigen
