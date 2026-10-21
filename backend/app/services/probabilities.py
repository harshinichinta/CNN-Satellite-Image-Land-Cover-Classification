from __future__ import annotations

import numpy as np


def apply_temperature(probabilities: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Calibration temperature must be positive.")

    scores = np.asarray(probabilities, dtype=np.float64)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("Expected a finite batch of class probability scores.")

    logits = np.log(np.clip(scores, 1e-12, 1.0)) / temperature
    logits -= np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return (exp_logits / np.sum(exp_logits, axis=1, keepdims=True)).astype(np.float32)


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 2 or len(scores) != len(targets) or not len(targets):
        raise ValueError("Calibration probabilities and labels must have matching samples.")
    if not np.isfinite(scores).all() or np.any(targets < 0) or np.any(targets >= scores.shape[1]):
        raise ValueError("Calibration data contains invalid scores or labels.")

    temperatures = np.geomspace(0.25, 5.0, 191)
    log_scores = np.log(np.clip(scores, 1e-12, 1.0))
    losses = []
    for temperature in temperatures:
        logits = log_scores / temperature
        logits -= np.max(logits, axis=1, keepdims=True)
        log_normalizer = np.log(np.exp(logits).sum(axis=1))
        losses.append(np.mean(log_normalizer - logits[np.arange(len(targets)), targets]))
    return float(temperatures[int(np.argmin(losses))])
