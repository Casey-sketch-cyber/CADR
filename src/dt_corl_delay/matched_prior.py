"""Construct the predeclared reference law for the matched-prior control."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .delay_process import (
    CategoricalDelay,
    ChangepointDelay,
    DriftingCategoricalDelay,
    FixedDelay,
    MarkovSwitchingDelay,
    build_delay_process,
)


MATCHED_PRIOR_VARIANT = "fixed_prior_matched"
MATCHED_PRIOR_IQL_METHOD = "cadr_iql_ablation_fixed_prior_matched"


def matched_delay_prior(scenario: Mapping[str, Any], max_delay: int) -> list[float]:
    """Represent a static reference law on the estimator support 0..max_delay.

    This helper is used before evaluation. Only its static probability vector
    is supplied to the controller. No channel state or sampled delay is read.
    """
    if max_delay < 0:
        raise ValueError("max_delay must be non-negative")
    kind = scenario["type"]
    if kind not in {"fixed", "categorical", "changepoint", "drifting", "markov"}:
        raise ValueError(
            f"fixed_prior_matched has no predeclared reference for {kind!r} scenarios"
        )
    if kind == "markov" and scenario.get("initial_regime", 0) != 0:
        raise ValueError("the configured Markov channel starts in regime 0")
    process = build_delay_process(scenario)
    if isinstance(process, FixedDelay):
        delays, probabilities = (process.delay,), (1.0,)
    elif isinstance(process, DriftingCategoricalDelay):
        delays, probabilities = process.delays, process.start_probabilities
    else:
        if isinstance(process, ChangepointDelay):
            process = process.before
        if isinstance(process, MarkovSwitchingDelay):
            process = process.regimes[process.initial_regime]
        assert isinstance(process, CategoricalDelay)
        delays, probabilities = process.delays, process.probabilities

    prior = [0.0] * (max_delay + 1)
    for delay, probability in zip(delays, probabilities):
        if not 0 <= delay <= max_delay:
            raise ValueError(f"reference delay {delay} is outside 0..{max_delay}")
        if not math.isfinite(probability):
            raise ValueError("reference probabilities must be finite")
        prior[delay] += probability
    total = math.fsum(prior)
    return [probability / total for probability in prior]


def matched_delay_priors(config: Mapping[str, Any]) -> dict[str, list[float]]:
    """Resolve matched priors for a run, or no overrides for other variants."""
    if config.get("ablation_variant") != MATCHED_PRIOR_VARIANT:
        return {}
    if config.get("posterior_updates", True):
        raise ValueError("fixed_prior_matched requires posterior_updates=false")
    priors = {}
    for scenario in config["scenarios"]:
        name = str(scenario["name"])
        if name in priors:
            raise ValueError(f"duplicate scenario name: {name}")
        priors[name] = matched_delay_prior(scenario, int(config["max_delay"]))
    if not priors:
        raise ValueError("fixed_prior_matched requires at least one scenario")
    return priors
