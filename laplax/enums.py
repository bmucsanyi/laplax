"""Enums for Laplace approximations."""

from enum import StrEnum


class LossFn(StrEnum):
    MSE = "mse"
    CROSSENTROPY = "cross_entropy"


class CurvApprox(StrEnum):
    FULL = "full"
    DIAGONAL = "diagonal"
    LOW_RANK = "low_rank"
