"""Normalisation of user-supplied training and holdout input.

The API accepts raw fan vibration/current features and standardises each
column with statistics from the **training** sample only (zero mean, unit
variance); constant columns become zero.  A bias column of ones is appended,
so the learned vector is ``[w_1 .. w_d, b]``.  Labels must be 0 (healthy) or
1 (shutdown warning).

Held-out records are standardised with the training sample's mean and standard
deviation, never with their own statistics, so the evaluation cannot leak
information from the holdout set into the model or its preprocessing.

The float64 design matrices are persisted, so any worker resumes on
bit-identical inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_RECORDS = 8
MAX_RECORDS = 64
MIN_HOLDOUT = 2
MAX_HOLDOUT = 16
MIN_EPOCHS = 2
MAX_EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 0.1
MOMENTUM = 0.9


class InputError(ValueError):
    """Invalid training submission."""


@dataclass(frozen=True)
class Standardisation:
    """Column statistics fitted on the training sample."""

    mean: np.ndarray
    std: np.ndarray
    constant: np.ndarray

    def as_json(self) -> dict[str, list]:
        return {
            "mean": [float(value) for value in self.mean],
            "std": [float(value) for value in self.std],
            "constant": [bool(value) for value in self.constant],
        }


def _validated_matrix(
    features: list[list[float]],
    labels: list[int],
    minimum: int,
    maximum: int,
    what: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not (minimum <= len(features) <= maximum):
        raise InputError(f"submit between {minimum} and {maximum} {what} records")
    if len(labels) != len(features):
        raise InputError(
            f"{what} features and labels must contain the same number of records"
        )
    if any(len(row) == 0 for row in features):
        raise InputError("each record needs at least one feature")
    if any(len(row) != len(features[0]) for row in features):
        raise InputError("every record must have the same number of features")

    try:
        matrix = np.asarray(features, dtype=np.float64)
        targets = np.asarray(labels, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{what} features and labels must be numeric") from exc
    if matrix.ndim != 2:
        raise InputError(f"{what} features must be a two-dimensional list of numbers")
    if targets.ndim != 1:
        raise InputError(f"{what} labels must be a flat list")
    if not np.all(np.isfinite(matrix)):
        raise InputError(f"{what} features must be finite numbers")
    if not np.all((targets == 0.0) | (targets == 1.0)):
        raise InputError(f"{what} labels must be 0 or 1")
    return matrix, targets


def _fit(matrix: np.ndarray) -> Standardisation:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    return Standardisation(
        mean=mean, std=std, constant=np.asarray(std < 1e-12, dtype=bool)
    )


def _apply(matrix: np.ndarray, stats: Standardisation) -> np.ndarray:
    scaled = (matrix - stats.mean) / np.where(stats.constant, 1.0, stats.std)
    scaled[:, stats.constant] = 0.0
    bias = np.ones((scaled.shape[0], 1), dtype=np.float64)
    return np.concatenate([scaled, bias], axis=1)


def normalise_with_stats(
    features: list[list[float]], labels: list[int]
) -> tuple[np.ndarray, np.ndarray, Standardisation]:
    """Validate the training sample, fit its statistics and transform it."""
    matrix, targets = _validated_matrix(
        features, labels, MIN_RECORDS, MAX_RECORDS, "training"
    )
    stats = _fit(matrix)
    return _apply(matrix, stats), targets, stats


def normalise(
    features: list[list[float]], labels: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    design, targets, _ = normalise_with_stats(features, labels)
    return design, targets


def normalise_holdout(
    features: list[list[float]],
    labels: list[int],
    stats: Standardisation,
    feature_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate held-out records and transform them with *training* statistics.

    ``feature_count`` is the raw (pre-bias) number of feature columns in the
    training sample.  The holdout set must contain at least one record of each
    class, because recall and false-positive rate are otherwise undefined.
    """
    matrix, targets = _validated_matrix(
        features, labels, MIN_HOLDOUT, MAX_HOLDOUT, "holdout"
    )
    if matrix.shape[1] != feature_count:
        raise InputError(
            f"holdout records must have the same {feature_count} features as training"
        )
    if not np.any(targets == 0.0) or not np.any(targets == 1.0):
        raise InputError(
            "holdout records must contain both healthy (0) and shutdown (1) labels"
        )
    return _apply(matrix, stats), targets
