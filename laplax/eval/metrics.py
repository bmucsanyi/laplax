"""Regression and classification metrics for evaluating uncertainty quantification.

This module provides a comprehensive suite of classification and regression metrics for
evaluating probabilistic models.

Key features include:
- **Classification Metrics**: Accuracy, top-k accuracy, cross-entropy, and
    multiclass Brier score.
- **Regression Metrics**: Root mean squared error (RMSE), q-value, and negative
    log-likelihood (NLL) for Gaussian distributions.
- **Bin Metrics**: Confidence and correctness metrics binned by confidence intervals.

The module leverages JAX for efficient numerical computation and supports flexible
evaluation for diverse model outputs.
"""

import math

import jax
import jax.numpy as jnp
from jax import lax

from laplax.types import Array, Float

# --------------------------------------------------------------------------------
# Classification metrics
# --------------------------------------------------------------------------------


def correctness(pred: Array, target: Array) -> Array:
    """Determine if each target label matches the top-1 prediction.

    Computes a binary indicator for whether the predicted class matches the
    target class. If the target is a 2D array, it is first reduced to its
    class index using `argmax`.

    Args:
        pred: Array of predictions with shape `(batch_size, num_classes)`.
        target: Array of ground truth labels, either 1D (class indices) or
            2D (one-hot encoded).

    Returns:
        Boolean array of shape `(batch_size,)` indicating correctness
        for each prediction.
    """
    pred = jnp.argmax(pred, axis=-1)

    if target.ndim == 2:
        target = jnp.argmax(target, axis=-1)

    return pred == target


def accuracy(pred: Array, target: Array, top_k: tuple[int] = (1,)) -> list[Array]:
    """Compute top-k accuracy for specified values of k.

    For each k in `top_k`, this function calculates the fraction of samples
    where the ground truth label is among the top-k predictions. If the target
    is a 2D array, it is reduced to its class index using `argmax`.

    Args:
        pred: Array of predictions with shape `(batch_size, num_classes)`.
        target: Array of ground truth labels, either 1D (class indices) or
            2D (one-hot encoded).
        top_k: Tuple of integers specifying the values of k for top-k accuracy.

    Returns:
        A list of accuracies corresponding to each k in `top_k`,
        expressed as percentages.
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


def cross_entropy(prob_p: Array, prob_q: Array, axis: int = -1) -> Array:
    """Compute cross-entropy between two probability distributions.

    This function calculates the cross-entropy of `prob_p` relative to `prob_q`,
    summing over the specified axis.

    Args:
        prob_p: Array of true probability distributions.
        prob_q: Array of predicted probability distributions.
        axis: Axis along which to compute the cross-entropy (default: -1).

    Returns:
        Cross-entropy values for each sample.
    """
    p_log_q = jax.scipy.special.xlogy(prob_p, prob_q)

    return -p_log_q.sum(axis=axis)


def multiclass_brier(prob: Array, target: Array) -> Array:
    """Compute the multiclass Brier score.

    The Brier score is a measure of the accuracy of probabilistic predictions.
    For multiclass classification, it calculates the mean squared difference
    between the predicted probabilities and the true target.

    Args:
        prob: Array of predicted probabilities with shape `(batch_size, num_classes)`.
        target: Array of ground truth labels, either 1D (class indices) or
            2D (one-hot encoded).

    Returns:
        Mean Brier score across all samples.
    """
    if target.ndim == 1:
        target = jax.nn.one_hot(target, num_classes=prob.shape[-1])

    preds_squared_sum = jnp.sum(prob**2, axis=-1, keepdims=True)
    score_components = 1 - 2 * prob + preds_squared_sum

    return -jnp.mean(target * score_components)


def calculate_bin_metrics(
    confidence: Array,
    correctness: Array,
    num_bins: int = 15,
) -> tuple[Array, Array, Array]:
    """Calculate bin-wise metrics for confidence and correctness.

    Computes the proportion of samples, average confidence, and average accuracy
    within each bin, where the bins are defined by evenly spaced confidence
    intervals.

    Args:
        confidence: Array of predicted confidence values with shape `(n,)`.
        correctness: Array of correctness labels (0 or 1) with shape `(n,)`.
        num_bins: Number of bins for dividing the confidence range (default: 15).

    Returns:
        Tuple of arrays:
            - Bin proportions: Proportion of samples in each bin.
            - Bin confidences: Average confidence for each bin.
            - Bin accuracies: Average accuracy for each bin.
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


def estimate_q(
    pred_mean: Array,
    pred_std: Array,
    target: Array,
    **kwargs,
) -> Float:
    r"""Estimate the q-value for predictions.

    The q-value is a measure of the squared error normalized by the predicted
    variance.

    Mathematically:
    $q = \frac{1}{n} \sum_{i=1}^n \frac{(y_i - \hat{y}_i)^2}{\sigma_i^2}$.

    Args:
        pred_mean: Array of predicted means.
        pred_std: Array of predicted standard deviations.
        target: Array of ground truth labels.
        **kwargs: Additional arguments (ignored).

    Returns:
        The estimated q-value.
    """
    del kwargs
    return jnp.mean(jnp.power(pred_mean - target, 2) / jnp.power(pred_std, 2))


def estimate_rmse(pred_mean: Array, target: Array, **kwargs) -> Float:
    r"""Estimate the root mean squared error (RMSE) for predictions.

    Mathematically:
    $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$.

    Args:
        pred_mean: Array of predicted means.
        target: Array of ground truth labels.
        **kwargs: Additional arguments (ignored).

    Returns:
        The RMSE value.
    """
    del kwargs
    return jnp.sqrt(jnp.mean(jnp.power(pred_mean - target, 2)))


def estimate_true_rmse(pred: Array, target: Array, **kwargs) -> Float:
    """Estimate the 'true' RMSE for predictions.

    This function computes the RMSE directly from the predictions and targets,
    without additional variance or uncertainty modeling.

    Args:
        pred: Array of predicted values.
        target: Array of ground truth labels.
        **kwargs: Additional arguments (ignored).

    Returns:
        The RMSE value.
    """
    del kwargs
    return jnp.sqrt(jnp.mean(jnp.power(pred - target, 2)))


def nll_gaussian(
    pred: Array,
    pred_std: Array,
    target: Array,
    *,
    scaled: bool = True,
    **kwargs,
) -> Float:
    r"""Compute the negative log-likelihood (NLL) for a Gaussian distribution.

    The NLL quantifies how well the predictive distribution fits the data,
    assuming a Gaussian distribution characterized by `pred` (mean) and `pred_std`
    (standard deviation).

    Mathematically:
    $\text{NLL} = - \sum_{i=1}^n \log \left( \frac{1}{\sqrt{2\pi \sigma_i^2}}
    \exp \left( -\frac{(y_i - \hat{y}_i)^2}{2\sigma_i^2} \right) \right)$.

    Args:
        pred: Array of predicted means for the dataset.
        pred_std: Array of predicted standard deviations for the dataset.
        target: Array of ground truth labels for the dataset.
        scaled: Whether to normalize the NLL by the number of samples (default: True).
        **kwargs: Additional arguments (ignored).

    Returns:
        The computed NLL value.
    """
    del kwargs

    # Ensure input arrays are 1D and of the same shape
    if not (pred.shape == pred_std.shape == target.shape):
        msg = "Arrays must have the same shape"
        raise ValueError(msg)

    # Compute residuals
    residuals = pred - target

    # Compute negative log likelihood
    nll_list = jax.scipy.stats.norm.logpdf(residuals, scale=pred_std)
    nll = -1 * jnp.sum(nll_list)

    # Scale the result by the number of data points if `scaled` is True
    if scaled:
        nll /= math.prod(pred.shape)

    return nll


DEFAULT_REGRESSION_METRICS = {
    "rmse": estimate_rmse,
    "q": estimate_q,
    "nll": nll_gaussian,
}
