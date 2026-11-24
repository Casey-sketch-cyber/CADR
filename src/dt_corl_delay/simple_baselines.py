"""Resolve static, declared reference information for the T4 baselines."""

from __future__ import annotations

import math
from typing import Any, Mapping


SIMPLE_AGE_DECISION_RULES = frozenset({"in_order_age", "fixed_horizon_model"})
_NOMINAL_REFERENCE_DELAYS = (0, 1, 2)
_NOMINAL_REFERENCE_PROBABILITIES = (0.65, 0.25, 0.10)


def resolve_simple_baseline(
    config: Mapping[str, Any],
) -> tuple[str | None, dict[str, int]]:
    """Return the method suffix and the same nominal horizon for each scenario."""
    decision_rule = config.get("age_decision_rule")
    if decision_rule == "in_order_age":
        return "in_order_age", {}
    if decision_rule != "fixed_horizon_model":
        return None, {}

    rule = config.get("fixed_horizon_rule", "mean")
    if rule not in {"mean", "max_support"}:
        raise ValueError("fixed_horizon_rule must be 'mean' or 'max_support'")
    if rule == "mean":
        mean = math.fsum(
            delay * probability for delay, probability in
            zip(_NOMINAL_REFERENCE_DELAYS, _NOMINAL_REFERENCE_PROBABILITIES)
        )
        # The nominal mean is 0.45. The predeclared one-step floor prevents
        # the T6 dynamics baseline from collapsing to direct observation.
        horizon = max(1, math.floor(mean + 0.5))
    else:
        horizon = max(_NOMINAL_REFERENCE_DELAYS)
    if horizon > int(config["max_delay"]):
        raise ValueError("the nominal fixed horizon exceeds max_delay")
    horizons: dict[str, int] = {}
    for scenario in config["scenarios"]:
        name = str(scenario["name"])
        if name in horizons:
            raise ValueError(f"duplicate scenario name: {name}")
        horizons[name] = horizon
    if not horizons:
        raise ValueError("fixed_horizon_model requires at least one scenario")
    return f"fixed_horizon_model_{rule}", horizons
