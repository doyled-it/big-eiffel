"""Small statistical helpers shared across methods."""

from __future__ import annotations

import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Arguments:
        k: Number of successes.
        n: Number of trials.
        z: Normal quantile (default 1.96 for a 95% interval).

    Returns:
        (low, high) bounds of the interval, as proportions.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted quantile of ``values``.

    Arguments:
        values: Sample values.
        weights: Non-negative weights, same shape as values.
        q: Quantile in [0, 1].

    Returns:
        The weighted quantile.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w) - 0.5 * w
    cw /= np.sum(w)
    return float(np.interp(q, cw, v))


def effective_n(weights: np.ndarray) -> float:
    """Kish effective sample size for a set of weights."""
    w = np.asarray(weights, dtype=float)
    s = w.sum()
    return float(s * s / np.sum(w * w)) if s > 0 else 0.0
