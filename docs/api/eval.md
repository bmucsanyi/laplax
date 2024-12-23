# Evaluation Module

The evaluation module provides functionality for propagating weight-space uncertainty to output-space predictions and evaluating probabilistic predictions on datasets.

## Overview

### Pushforward Methods
Currently supported pushforward methods are:

- **Monte Carlo**: Ensemble-based predictions using weight samples
- **Linearized**: Efficient uncertainty propagation using model linearization
- **GP Kernel**: Posterior GP kernel.

Each method provides functions for computing predictive means, variances, and covariances in output space.

### Evaluation Utilities
The module includes utilities for:

- Evaluating predictions on datasets
- Computing metrics across datasets
- Managing metric computations and result aggregation
- Transforming evaluation results

## Key functions

All pushforward functions follow the pattern of the following setup function.

::: laplax.eval.pushforward.set_prob_predictive

Once the pushforward is setup, they can be applied and evaluated using the following shared functions:

::: laplax.eval.utils.evaluate_on_dataset

::: laplax.eval.utils.evaluate_metrics_on_dataset
