# Pushforward predictives

## Linearized Laplace

The linearized Laplace approximation propagates uncertainty through a first-order Taylor expansion of the model around the MAP estimate. The following functions implement this approach:

### Setup Functions

::: laplax.eval.pushforward.set_lin_pushforward


### Core Functions

::: laplax.eval.pushforward.lin_setup

::: laplax.eval.pushforward.lin_pred

::: laplax.eval.pushforward.lin_pred_var

::: laplax.eval.pushforward.lin_pred_std

::: laplax.eval.pushforward.lin_pred_cov

::: laplax.eval.pushforward.lin_n_samples_fn


### Posterior GP Kernel

::: laplax.eval.pushforward.set_posterior_gp_kernel


## Sample-based Laplace

The sample-based Laplace approximation propagates uncertainty by sampling from the weight-space posterior and evaluating the model at these samples. The following functions implement this approach:

### Setup Functions

::: laplax.eval.pushforward.set_mc_pushforward


### Core Functions

::: laplax.eval.pushforward.mc_setup

::: laplax.eval.pushforward.mc_pred_mean

::: laplax.eval.pushforward.mc_pred_var

::: laplax.eval.pushforward.mc_pred_std

::: laplax.eval.pushforward.mc_pred_cov

::: laplax.eval.pushforward.mc_samples
