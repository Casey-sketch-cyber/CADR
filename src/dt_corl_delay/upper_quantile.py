"""Validation, naming, and inverse CDF for the T10 age decision rule."""

from __future__ import annotations

from decimal import Decimal
import math
from numbers import Real
from typing import Any, Mapping

import numpy as np


def validate_age_quantile(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not 0 < value <= 1 or not math.isfinite(value):
        raise ValueError("upper_quantile requires a finite numeric age_quantile in (0, 1]")
    return float(value)


def upper_quantile_method(config: Mapping[str, Any]) -> str | None:
    if config.get("age_decision_rule") != "upper_quantile":
        return None
    alpha = validate_age_quantile(config.get("age_quantile"))
    percentage = format((Decimal(str(alpha)) * 100).normalize(), "f")
    return f"upper_quantile_a{percentage}"


def upper_age_quantile(probabilities: np.ndarray, alpha: float) -> int:
    """Return the first CDF index at or above alpha for the feasible age law.

    The law is truncated at the exact age or has positive prior mass through
    the last feasible age K. At alpha=1 use K explicitly because a rounded
    prefix CDF can reach one before the mathematically positive tail.
    """
    largest_age = probabilities.size - 1
    if alpha == 1.0:
        return largest_age
    cumulative = np.cumsum(probabilities, dtype=np.float64)
    return min(largest_age, int(np.searchsorted(cumulative, alpha, side="left")))
