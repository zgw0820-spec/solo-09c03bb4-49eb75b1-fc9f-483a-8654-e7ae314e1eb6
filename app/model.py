"""Standard logistic regression: sigmoid, cross-entropy, SGD + momentum.

The arithmetic is deliberately fixed and float64 so the same sequence of
updates can be replayed from a checkpoint.  A restart therefore continues from
the exact weight and velocity vectors the previous worker committed.
"""
from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable form: never exponentiate a large positive number.
    out = np.empty_like(z, dtype=np.float64)
    non_negative = z >= 0
    out[non_negative] = 1.0 / (1.0 + np.exp(-z[non_negative]))
    exp_z = np.exp(z[~non_negative])
    out[~non_negative] = exp_z / (1.0 + exp_z)
    return out


def batch_loss(weights: np.ndarray, features: np.ndarray, labels: np.ndarray) -> float:
    """Mean binary cross-entropy of one batch."""
    logits = features @ weights
    probabilities = sigmoid(logits)
    eps = 1e-12
    positive = labels * np.log(probabilities + eps)
    negative = (1.0 - labels) * np.log(1.0 - probabilities + eps)
    return float(np.mean(-(positive + negative)))


def sgd_step(
    weights: np.ndarray,
    velocity: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    learning_rate: float,
    momentum: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """One mini-batch update.

    Returns the new weights, the new momentum velocity and the loss measured on
    the batch *before* the update (so the reported loss corresponds to the
    parameters that were used to compute it).  The caller commits the resulting
    weights alongside that loss, keeping the checkpoint internally consistent.
    """
    probabilities = sigmoid(features @ weights)
    loss = batch_loss(weights, features, labels)
    gradient = features.T @ (probabilities - labels) / features.shape[0]
    velocity = momentum * velocity - learning_rate * gradient
    weights = weights + velocity
    return weights, velocity, loss


def dataset_loss(weights: np.ndarray, features: np.ndarray, labels: np.ndarray) -> float:
    return batch_loss(weights, features, labels)


PREDICTION_THRESHOLD = 0.5


def _rate(numerator: int, denominator: int) -> float | None:
    """A ratio, or ``None`` when its denominator is empty."""
    return None if denominator == 0 else numerator / denominator


def evaluate_holdout(
    weights: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    min_recall: float,
    max_false_positive_rate: float,
) -> dict:
    """Score held-out records once, with the final weights.

    A record is flagged (early warning) when its shutdown probability is at or
    above ``PREDICTION_THRESHOLD`` (0.5).

    * recall    = TP / (TP + FN) - share of shutdown records that were flagged
    * FPR       = FP / (FP + TN) - share of healthy records that were flagged
    * threshold_met = recall >= min_recall and FPR <= max_false_positive_rate

    The caller guarantees the holdout sample contains both classes, so both
    rates are defined.
    """
    probabilities = sigmoid(features @ weights)
    predicted = probabilities >= PREDICTION_THRESHOLD
    actual = labels >= 0.5
    true_positive = int(np.sum(predicted & actual))
    false_negative = int(np.sum(~predicted & actual))
    false_positive = int(np.sum(predicted & ~actual))
    true_negative = int(np.sum(~predicted & ~actual))
    positives = true_positive + false_negative
    negatives = false_positive + true_negative
    recall = _rate(true_positive, positives)
    false_positive_rate = _rate(false_positive, negatives)
    return {
        "evaluated": True,
        "threshold_met": bool(
            recall is not None
            and false_positive_rate is not None
            and recall >= min_recall
            and false_positive_rate <= max_false_positive_rate
        ),
        "reason": "holdout_evaluated",
        "prediction_threshold": PREDICTION_THRESHOLD,
        "holdout_records": int(labels.shape[0]),
        "positive_records": positives,
        "negative_records": negatives,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "min_recall": float(min_recall),
        "max_false_positive_rate": float(max_false_positive_rate),
    }

