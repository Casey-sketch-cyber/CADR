"""Sanity check for the observation-age posterior used by CADR.

Simulates the repository's DelayedObservationChannel (ack_lag=1) and compares,
on steps where the received observation changes, the empirical observation-age
distribution with
  (a) the manuscript's Eq.(9)/(10): unconditional CDF product, truncated, and
  (b) the hazard-based conditional posterior proposed in docs/revision/TASKS.md (T1).
The true packet-delay distribution q is given to both, so the comparison isolates
the posterior formula from delay-estimation error.

It also reports how often the age is exactly known (unchanged observation).

Usage (CPU only, a few seconds):
    python scripts/revision/age_posterior_check.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dt_corl_delay.channel import DelayedObservationChannel  # noqa: E402


class CategoricalDelay:
    def __init__(self, delays, probabilities):
        self.delays = list(delays)
        self.probabilities = list(probabilities)

    def reset(self) -> None:
        pass

    def sample(self, generated_step, rng):
        return rng.choices(self.delays, weights=self.probabilities)[0]


def legacy_cdf_posterior(q: np.ndarray, max_age: int) -> np.ndarray:
    cdf = np.cumsum(q)
    weights = np.zeros(len(q))
    missing = 1.0
    for k in range(len(q)):
        weights[k] = missing * cdf[k]
        missing *= 1.0 - cdf[k]
    weights = weights[: max_age + 1]
    return weights / weights.sum()


def hazard_posterior(q: np.ndarray, max_age: int) -> np.ndarray:
    survival = np.array([q[k:].sum() for k in range(len(q))])
    hazard = np.where(survival > 1e-15, q / np.maximum(survival, 1e-15), 1.0)
    weights = np.array(
        [hazard[k] * np.prod(1.0 - hazard[:k]) for k in range(max_age + 1)]
    )
    return weights / weights.sum()


def total_variation(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.abs(a - b).sum())


def run(delays, probabilities, steps=200_000, seed=0) -> None:
    q = np.zeros(max(delays) + 1)
    q[list(delays)] = probabilities
    channel = DelayedObservationChannel(
        CategoricalDelay(delays, probabilities), random.Random(seed), ack_lag=1
    )
    channel.reset(0, 0)
    previous_generation = 0
    unchanged = 0
    ages_by_max_age: dict[int, list[int]] = {}
    for t in range(1, steps):
        channel.send(t, t)
        generation = channel.advance(t).selected_packet.generated_step
        if generation == previous_generation:
            unchanged += 1
        else:
            # With ack_lag=1, the latest revealed generation equals the
            # generation selected at t-1.
            max_age = t - previous_generation - 1
            ages_by_max_age.setdefault(max_age, []).append(t - generation)
        previous_generation = generation

    print(f"delays={list(delays)} probs={list(probabilities)}")
    print(f"  age exactly known (unchanged observation): {unchanged / steps:.2f}")
    for max_age in sorted(ages_by_max_age)[:4]:
        ages = ages_by_max_age[max_age]
        empirical = np.bincount(ages, minlength=max_age + 1)[: max_age + 1] / len(ages)
        legacy = legacy_cdf_posterior(q, max_age)
        hazard = hazard_posterior(q, max_age)
        print(
            f"  max_age={max_age:2d} n={len(ages):6d} "
            f"TV(legacy Eq.10)={total_variation(legacy, empirical):.3f} "
            f"TV(hazard)={total_variation(hazard, empirical):.3f}"
        )


if __name__ == "__main__":
    run([0, 1, 2], [0.65, 0.25, 0.10])
    run([3, 5, 7], [0.20, 0.45, 0.35])
    run([8, 12, 16], [0.20, 0.45, 0.35])
