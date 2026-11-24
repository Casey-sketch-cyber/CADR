"""Online delay inference from delayed ACKs and right-censored packets.

The estimator never receives the age of the observation used for the current
decision.  It only uses ACKs revealed by the channel and the fact that one
sensor packet is generated after every control decision.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .channel import Ack


def hazard_age_probabilities(
    delay_probabilities: np.ndarray, maximum_age: int
) -> np.ndarray:
    """Return the age posterior given an observation change and an age bound.

    If newer packets had not arrived before this step, each can arrive now
    with discrete hazard q(k) / P(delay >= k). The freshest of those arrivals
    determines the age. Condition its probabilities on an arrival by age K.
    The survival factors telescope, so the normalized result is q restricted
    to feasible ages. If q has no mass there, return a uniform distribution.
    """
    probabilities, _ = _hazard_age_probabilities_with_fallback(
        delay_probabilities, maximum_age
    )
    return probabilities


def _hazard_age_probabilities_with_fallback(
    delay_probabilities: np.ndarray, maximum_age: int
) -> tuple[np.ndarray, bool]:
    q = np.asarray(delay_probabilities, dtype=np.float64)
    if q.ndim != 1 or q.size == 0 or not np.isfinite(q).all() or np.any(q < 0.0):
        raise ValueError("delay_probabilities must be a finite, non-negative vector")
    if not 0 <= maximum_age < q.size:
        raise ValueError("maximum_age is outside the delay support")
    if not np.any(q > 0.0):
        raise ValueError("delay_probabilities must have positive mass")
    if not np.any(q[: maximum_age + 1] > 0.0):
        return np.full(maximum_age + 1, 1.0 / (maximum_age + 1)), True
    if maximum_age == 0:
        # A change with only one feasible age determines that age.
        return np.asarray([1.0], dtype=np.float64), False

    q = q / q.max()
    survival = np.cumsum(q[::-1])[::-1][: maximum_age + 1]
    weights = np.divide(
        q[: maximum_age + 1],
        survival,
        out=np.ones_like(survival),
        where=survival > 0.0,
    )
    # S(k+1)/S(k) is 1 - h(k) and avoids cancellation for hazards near one.
    not_arrived = np.divide(
        survival[1:],
        survival[:-1],
        out=np.zeros_like(survival[:-1]),
        where=survival[:-1] > 0.0,
    )
    weights[1:] *= np.cumprod(not_arrived)
    mass = float(weights.sum())
    if mass == 0.0:
        # The full-law rescaling can underflow when feasible positive mass
        # is tiny compared to tail mass. Normalize the feasible law directly.
        feasible = np.asarray(delay_probabilities, dtype=np.float64)[: maximum_age + 1]
        feasible = feasible / feasible.max()
        return feasible / feasible.sum(), False
    return weights / mass, False


def validate_age_posterior(
    age_posterior: str, ack_lag: int, use_content_conditioning: bool
) -> None:
    if age_posterior not in {"legacy_cdf", "hazard"}:
        raise ValueError("age_posterior must be 'legacy_cdf' or 'hazard'")
    if age_posterior == "hazard" and use_content_conditioning and ack_lag != 1:
        raise ValueError("the content-conditioned hazard posterior requires ack_lag=1")


class ContentConditionedAgePosterior:
    """Track the ACK and observation history shared by CADR and hazard DT-ACK."""

    def __init__(self, max_delay: int, age_posterior: str) -> None:
        self.max_delay = max_delay
        self.age_posterior = age_posterior
        self._known_selected_generated_step = 0
        self._previous_observation: np.ndarray | None = None
        self.hazard_zero_mass_fallback = False

    def reset(self, initial_observation: np.ndarray) -> None:
        self._known_selected_generated_step = 0
        self._previous_observation = np.asarray(initial_observation).copy()
        self.hazard_zero_mass_fallback = False

    def probabilities(
        self,
        observation: np.ndarray,
        decision_step: int,
        acknowledgements: Sequence[Ack],
        base_probabilities: np.ndarray,
        delay_probabilities: np.ndarray,
    ) -> tuple[np.ndarray, bool, bool]:
        self.hazard_zero_mass_fallback = False
        for acknowledgement in acknowledgements:
            self._known_selected_generated_step = max(
                self._known_selected_generated_step,
                int(acknowledgement.generated_step),
            )
        changed = (
            self._previous_observation is not None
            and not np.array_equal(observation, self._previous_observation)
        )
        exact_age = False
        if decision_step == 0:
            probabilities = np.asarray([1.0], dtype=np.float64)
            exact_age = True
        elif not changed:
            age = min(
                self.max_delay,
                max(0, decision_step - self._known_selected_generated_step),
            )
            probabilities = np.zeros(age + 1, dtype=np.float64)
            probabilities[age] = 1.0
            exact_age = True
        else:
            # A changed observation is newer than the latest revealed packet.
            maximum_possible_age = min(
                len(base_probabilities) - 1,
                max(0, decision_step - self._known_selected_generated_step - 1),
            )
            if self.age_posterior == "hazard":
                probabilities, self.hazard_zero_mass_fallback = (
                    _hazard_age_probabilities_with_fallback(
                        delay_probabilities, maximum_possible_age
                    )
                )
            else:
                probabilities = base_probabilities[: maximum_possible_age + 1].copy()
                mass = float(probabilities.sum())
                if mass <= 1e-15:
                    probabilities = np.full(
                        maximum_possible_age + 1,
                        1.0 / (maximum_possible_age + 1),
                        dtype=np.float64,
                    )
                else:
                    probabilities /= mass
        self._previous_observation = observation.copy()
        return probabilities, exact_age, changed


class CensoredAckDelayEstimator:
    """Exponentially weighted discrete delay estimator with censoring.

    At decision ``t``, a missing ACK for the packet generated at ``g`` implies
    ``delay > t - g - ack_lag``.  An EM update uses that survival information
    rather than incorrectly treating recently missing ACKs as short delays.
    With ``use_censoring=False``, only revealed delays and the prior contribute.
    """

    def __init__(
        self,
        max_delay: int,
        ack_lag: int = 1,
        memory_steps: int = 128,
        decay_time: float = 48.0,
        prior_strength: float = 2.0,
        prior_probabilities: Sequence[float] | None = None,
        em_iterations: int = 8,
        use_censoring: bool = True,
    ) -> None:
        if max_delay < 0:
            raise ValueError("max_delay must be non-negative")
        if ack_lag < 1:
            raise ValueError("ack_lag must be at least one")
        if memory_steps < max_delay + ack_lag + 1:
            raise ValueError("memory_steps must cover max_delay and ack_lag")
        if decay_time <= 0.0 or prior_strength <= 0.0:
            raise ValueError("decay_time and prior_strength must be positive")
        if em_iterations < 1:
            raise ValueError("em_iterations must be positive")

        self.max_delay = max_delay
        self.ack_lag = ack_lag
        self.memory_steps = memory_steps
        self.decay_time = decay_time
        self.prior_strength = prior_strength
        self.em_iterations = em_iterations
        self.use_censoring = use_censoring
        if prior_probabilities is None:
            # A weak short-delay preference makes the cold start usable while
            # retaining non-zero support for every OOD delay in the model.
            support = np.arange(max_delay + 1, dtype=np.float64)
            prior = np.exp(-support / max(1.0, max_delay / 3.0))
        else:
            prior = np.asarray(prior_probabilities, dtype=np.float64)
            if prior.shape != (max_delay + 1,):
                raise ValueError("prior_probabilities has the wrong length")
        if not np.isfinite(prior).all() or np.any(prior <= 0.0):
            raise ValueError("prior probabilities must be finite and positive")
        self._prior = prior / prior.sum()
        self._probabilities = self._prior.copy()
        self._exact_delays: dict[int, int] = {}
        self._last_step = -1

    def reset(self) -> None:
        self._probabilities = self._prior.copy()
        self._exact_delays.clear()
        self._last_step = -1

    @property
    def probabilities(self) -> np.ndarray:
        return self._probabilities.copy()

    @property
    def entropy(self) -> float:
        probabilities = np.clip(self._probabilities, 1e-12, 1.0)
        return float(-(probabilities * np.log(probabilities)).sum())

    def update(self, decision_step: int, acknowledgements: Sequence[Ack]) -> np.ndarray:
        if decision_step <= self._last_step:
            raise ValueError("decision steps must be strictly increasing")
        self._last_step = decision_step
        for acknowledgement in acknowledgements:
            if acknowledgement.revealed_step > decision_step:
                raise ValueError("an ACK cannot be used before its reveal step")
            delay = int(acknowledgement.delay)
            if delay < 0 or delay > self.max_delay:
                raise ValueError(
                    f"observed delay {delay} is outside configured support 0..{self.max_delay}"
                )
            self._exact_delays[int(acknowledgement.generated_step)] = delay

        first_generated = max(1, decision_step - self.memory_steps + 1)
        relevant_steps = range(first_generated, decision_step + 1)
        probabilities = self._probabilities.copy()
        support = np.arange(self.max_delay + 1)
        # Without latent delays, one pass gives the complete weighted counts.
        iterations = self.em_iterations if self.use_censoring else 1
        for _ in range(iterations):
            counts = self.prior_strength * self._prior
            for generated_step in relevant_steps:
                age = decision_step - generated_step
                weight = math.exp(-age / self.decay_time)
                exact_delay = self._exact_delays.get(generated_step)
                if exact_delay is not None:
                    counts[exact_delay] += weight
                    continue
                if not self.use_censoring:
                    continue

                censoring_threshold = age - self.ack_lag
                if censoring_threshold < 0:
                    continue
                allowed = support > censoring_threshold
                mass = float(probabilities[allowed].sum())
                if mass <= 1e-15:
                    # This can only happen under severe model mismatch. Keep a
                    # small amount of support at the configured upper bound.
                    counts[-1] += weight
                else:
                    counts[allowed] += weight * probabilities[allowed] / mass
            probabilities = counts / counts.sum()
        self._probabilities = probabilities

        # Entries older than the active likelihood window cannot affect future
        # estimates and are discarded to keep deployment memory bounded.
        stale = [step for step in self._exact_delays if step < first_generated]
        for generated_step in stale:
            del self._exact_delays[generated_step]
        return self.probabilities

    def effective_age_probabilities(self, decision_step: int) -> np.ndarray:
        """Convert packet-delay probabilities into current observation ages.

        The freshest received packet has age ``a`` when its delay is no more
        than ``a`` and every newer packet has not arrived.  Probability beyond
        the episode's current age is folded into the oldest feasible state.
        """

        if decision_step < 0:
            raise ValueError("decision_step must be non-negative")
        maximum_age = min(self.max_delay, decision_step)
        cdf = np.cumsum(self._probabilities)
        full = np.zeros(self.max_delay + 1, dtype=np.float64)
        newer_missing = 1.0
        for age in range(self.max_delay + 1):
            full[age] = newer_missing * cdf[age]
            newer_missing *= 1.0 - cdf[age]
        # Numerical residue and ages impossible near reset belong to the
        # oldest feasible initial observation.
        result = full[: maximum_age + 1].copy()
        if maximum_age < self.max_delay:
            result[-1] += full[maximum_age + 1 :].sum()
        result[-1] += max(0.0, 1.0 - float(full.sum()))
        result /= result.sum()
        return result
