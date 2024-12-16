"""Plotting utilities."""

import matplotlib.pyplot as plt


def plot_regression_with_uncertainty(
    train_input,
    train_target,
    X_grid,
    Y_pred,
    Y_var,
    title="Model Predictions with Uncertainty",
    xlabel="Input",
    ylabel="Target",
):
    """Plot training data, model predictions, and uncertainty.

    Args:
        train_input: Training inputs (e.g., X_train).
        train_target: Training targets (e.g., y_train).
        X_grid: Input points for predictions.
        Y_pred: Predicted mean values.
        Y_var: Predicted variance (for uncertainty bounds).
        title: Plot title (default: "Model Predictions with Uncertainty").
        xlabel: Label for the x-axis (default: "Input").
        ylabel: Label for the y-axis (default: "Target").
    """
    _fig, ax = plt.subplots(figsize=(8, 6))

    # Plot training points
    ax.scatter(
        train_input,
        train_target,
        label="Training Points",
        color="green",
        s=50,
        edgecolor="k",
    )

    # Plot predicted mean
    ax.plot(X_grid, Y_pred, label="Prediction Mean", color="blue", linewidth=2)

    # Add uncertainty band
    ax.fill_between(
        X_grid[:, 0],
        Y_pred - 1.96 * Y_var,
        Y_pred + 1.96 * Y_var,
        color="cornflowerblue",
        alpha=0.3,
        label="Confidence Interval (95%)",
    )

    # Customize plot appearance
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10, frameon=True, shadow=True)

    # Show the plot
    plt.show()
