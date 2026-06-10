"""Statistical tests for comparing dispatching rules."""

from __future__ import annotations

from typing import List, Tuple


def wilcoxon_test(
    at_baseline: List[float],
    at_proposed: List[float],
) -> Tuple[float, float]:
    """Wilcoxon signed-rank test: baseline vs proposed (paired by seed).

    Returns
    -------
    statistic : float
    p_value   : float
    """
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(at_baseline, at_proposed, alternative="greater")
        return float(stat), float(p)
    except ImportError:
        return _manual_wilcoxon(at_baseline, at_proposed)
    except ValueError:
        # All differences zero → no test
        return 0.0, 1.0


def _manual_wilcoxon(xs: List[float], ys: List[float]) -> Tuple[float, float]:
    """Minimal implementation when scipy is unavailable."""
    diffs = [x - y for x, y in zip(xs, ys)]
    diffs = [d for d in diffs if abs(d) > 1e-12]
    if not diffs:
        return 0.0, 1.0

    abs_diffs = sorted(enumerate(abs(d) for d in diffs), key=lambda x: x[1])
    ranks = {i: r + 1 for r, (i, _) in enumerate(abs_diffs)}

    w_plus = sum(ranks[i] for i, d in enumerate(diffs) if d > 0)
    n = len(diffs)
    # Approximate p-value using normal approximation
    import math
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if sigma == 0:
        return w_plus, 1.0
    z = (w_plus - mu) / sigma
    # one-sided p-value (greater)
    p = 1 - _normal_cdf(z)
    return w_plus, p


def _normal_cdf(z: float) -> float:
    import math
    return (1.0 + math.erf(z / math.sqrt(2))) / 2
