# noqa: D100
import math

import jax
import jax.numpy as jnp
from jax import lax

# --------------------------------------------------------------------------------
# Classification metrics
# --------------------------------------------------------------------------------


def correctness(pred: jax.Array, target: jax.Array) -> jax.Array:
    """Compute whether each target label is the top-1 prediction of the output.

    If target is a 2D array, its argmax is taken before the calculation.
    """
    pred = jnp.argmax(pred, axis=-1)

    if target.ndim == 2:
        target = jnp.argmax(target, axis=-1)

    return pred == target


def accuracy(
    pred: jax.Array, target: jax.Array, top_k: tuple[int] = (1,)
) -> list[jax.Array]:
    """Compute the accuracy over the k top predictions for the specified values of k.

    If target is a 2D array, its argmax is taken before the calculation.
    """
    max_k = min(max(top_k), pred.shape[1])
    batch_size = target.shape[0]

    _, pred = lax.top_k(pred, max_k)
    pred = pred.T

    if target.ndim == 2:
        target = jnp.argmax(target, axis=-1)

    correctness = pred == target.reshape(1, -1)

    return [
        jnp.sum(correctness[: min(k, max_k)].reshape(-1).astype(jnp.float32))
        * 100.0
        / batch_size
        for k in top_k
    ]


def cross_entropy(prob_p: jax.Array, prob_q: jax.Array, axis: int = -1) -> jax.Array:
    p_log_q = jax.scipy.special.xlogy(prob_p, prob_q)

    return -p_log_q.sum(axis=axis)


def multiclass_brier(prob: jax.Array, target: jax.Array) -> jax.Array:
    if target.ndim == 1:
        target = jax.nn.one_hot(target, num_classes=prob.shape[-1])

    preds_squared_sum = jnp.sum(prob**2, axis=-1, keepdims=True)
    score_components = 1 - 2 * prob + preds_squared_sum

    return -jnp.mean(target * score_components)


def calculate_bin_metrics(
    confidence: jax.Array,
    correctness: jax.Array,
    num_bins: int = 15,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Calculate the binwise accuracies, confidences and proportions of samples.

    Args:
    ----
        confidence: Float tensor of shape (n,) containing predicted confidences.
        correctness: Float tensor of shape (n,) containing the true correctness
            labels.
        num_bins: Number of equally sized bins.

    Returns:
    -------
        bin_proportions: Float tensor of shape (num_bins,) containing proportion
            of samples in each bin. Sums up to 1.
        bin_confidences: Float tensor of shape (num_bins,) containing the average
            confidence for each bin.
        bin_accuracies: Float tensor of shape (num_bins,) containing the average
            accuracy for each bin.

    """
    bin_boundaries = jnp.linspace(0, 1, num_bins + 1)
    indices = jnp.digitize(confidence, bin_boundaries) - 1
    indices = jnp.clip(indices, min=0, max=num_bins - 1)

    bin_counts = jnp.zeros(num_bins, dtype=confidence.dtype)
    bin_confidences = jnp.zeros(num_bins, dtype=confidence.dtype)
    bin_accuracies = jnp.zeros(num_bins, dtype=correctness.dtype)

    bin_counts = bin_counts.at[indices].add(1)
    bin_confidences = bin_confidences.at[indices].add(confidence)
    bin_accuracies = bin_accuracies.at[indices].add(correctness)

    bin_proportions = bin_counts / bin_counts.sum()
    pos_counts = bin_counts > 0
    bin_confidences = jnp.where(pos_counts, bin_confidences / bin_counts, 0)
    bin_accuracies = jnp.where(pos_counts, bin_accuracies / bin_counts, 0)

    return bin_proportions, bin_confidences, bin_accuracies


# --------------------------------------------------------------------------------
# Regression metrics
# --------------------------------------------------------------------------------

# REGRESSION METRICS


def estimate_q(
    pred_mean: jnp.ndarray,
    pred_std: jnp.ndarray,
    target: jnp.ndarray,
) -> float:
    """Estimate the q value."""
    return jnp.mean(jnp.power(pred_mean - target, 2) / jnp.power(pred_std, 2))


def estimate_rmse(pred_mean: jax.Array, target: jax.Array) -> float:
    """Estimate the RMSE."""
    return jnp.sqrt(jnp.mean(jnp.power(pred_mean - target, 2)))


def nll_gaussian(  # noqa: D417
    pred: jnp.ndarray,
    pred_std: jnp.ndarray,
    target: jnp.ndarray,
    scaled: bool = True,  # noqa: FBT001, FBT002
    **kwargs,
) -> float:
    """Negative log likelihood for a Gaussian distribution in JAX.

    The negative log likelihood for held out data (y_true) given predictive
    uncertainty with mean (y_pred) and standard-deviation (y_std).

    Args:
        pred: 1D array of the predicted means for the held out dataset.
        pred_std: 1D array of the predicted standard deviations for the held out
            dataset.
        target: 1D array of the true labels in the held out dataset.
        scaled: Whether to scale the negative log likelihood by the size
            of the held out set.

    Returns:
        The negative log likelihood for the held out set.

    This code follows: https://github.com/uncertainty-toolbox/uncertainty-toolbox/blob/main/uncertainty_toolbox/metrics_scoring_rule.py
    """
    # Ensure input arrays are 1D and of the same shape
    assert (  # noqa: S101
        pred.shape == pred_std.shape == target.shape
    ), "Arrays must have the same shape."

    # Compute residuals
    residuals = pred - target

    # Compute negative log likelihood
    nll_list = jax.scipy.stats.norm.logpdf(residuals, scale=pred_std)
    nll = -1 * jnp.sum(nll_list)

    # Scale the result by the number of data points if `scaled` is True
    if scaled:
        nll /= math.prod(pred.shape)

    return nll
