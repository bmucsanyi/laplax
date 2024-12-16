# import jax
# from typing import Callable, Tuple

# def laplace(
#         model_fn: Callable,
#         params: jax.PyTreeLike,
#         data: Tuple[jax.Array, jax.Array],
# ):
#     """Get the Laplace approximation for a model.

#     Args:
#         model_fn: A function that computes the log likelihood of the model.
#         params: The parameters of the model.
#         data: The data to fit the model to.

#     Returns:
#         The Laplace approximation.
#     """
