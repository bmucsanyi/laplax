from collections.abc import Iterator  # noqa: D100

import jax.numpy as jnp
import numpy as np
from jax import random


# Function to create the sinusoid dataset
def get_sinusoid_example(
    n_data: int = 150,
    sigma_noise: float = 0.3,
    batch_size: int = 150,
    rng_key=None,
) -> tuple[
    jnp.ndarray, jnp.ndarray, Iterator[tuple[jnp.ndarray, jnp.ndarray]], jnp.ndarray
]:
    if rng_key is None:
        rng_key = random.PRNGKey(0)
    """Generate a sinusoid dataset.

    Args:
        n_data: Number of data points.
        sigma_noise: Standard deviation of the noise.
        batch_size: Batch size for the data loader.
        rng_key: Random number generator key.

    Returns:
        X_train: Training input data.
        y_train: Training target data.
        train_loader: Data loader for training data.
        X_test: Testing input data.
    """
    # Split RNG key for reproducibility
    rng_key, rng_noise = random.split(rng_key)

    # Generate random training data
    X_train = random.uniform(rng_key, (n_data, 1)) * 8  # X_train values between 0 and 8
    noise = random.normal(rng_noise, X_train.shape) * sigma_noise
    y_train = jnp.sin(X_train) + noise

    # Create a simple data loader function (generator)
    def _data_loader(X, y, batch_size):
        dataset_size = X.shape[0]
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)  # noqa: NPY002
        for start_idx in range(0, dataset_size, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            yield X[batch_indices], y[batch_indices]

    # Generate testing data
    X_test = jnp.linspace(-5, 13, 500).reshape(-1, 1)

    # Create the training data loader
    train_loader = _data_loader(X_train, y_train, batch_size)

    return X_train, y_train, train_loader, X_test
