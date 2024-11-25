"""Baseline for Laplace approximation.

This files contain implementations for input and weight space perturbations.
The design principle is given by the following structure:

input perturbations:
- model_fn : x -> model(x, p)
- get_mean : x -> x
- get_cov_scale : tau -> tau * Id
- get_prob_predictive : x -> model(x + eps, p) for eps ~ N(0, tau * Id)

"""

# --------------------------------------------------------------------
# BASELINE: Input perturbations
# --------------------------------------------------------------------


# --------------------------------------------------------------------
# BASELINE: Weight space perturbations
# --------------------------------------------------------------------
