from __future__ import annotations

import unittest

import numpy as np

from dt_corl_delay.channel import Ack
from dt_corl_delay.delay_belief import CensoredAckDelayEstimator


def _ack(generated_step: int, delay: int, ack_lag: int = 1) -> Ack:
    delivered = generated_step + delay
    return Ack(
        packet_id=generated_step,
        generated_step=generated_step,
        delivered_step=delivered,
        revealed_step=delivered + ack_lag,
        delay=delay,
    )


class CensoredAckDelayEstimatorTest(unittest.TestCase):
    def test_fixed_delay_converges_without_same_step_leakage(self) -> None:
        estimator = CensoredAckDelayEstimator(max_delay=6, prior_strength=0.2)
        pending: dict[int, list[Ack]] = {}
        for generated_step in range(1, 81):
            acknowledgement = _ack(generated_step, delay=3)
            pending.setdefault(acknowledgement.revealed_step, []).append(acknowledgement)
        for decision_step in range(81):
            estimator.update(decision_step, pending.get(decision_step, ()))
        self.assertGreater(estimator.probabilities[3], 0.95)
        self.assertGreater(estimator.effective_age_probabilities(80)[3], 0.90)

    def test_missing_ack_is_treated_as_right_censored(self) -> None:
        estimator = CensoredAckDelayEstimator(
            max_delay=5,
            prior_strength=0.1,
            prior_probabilities=np.ones(6),
        )
        estimator.update(0, ())
        estimator.update(1, ())
        estimator.update(2, ())
        # Packet g=1 has no ACK at t=2, so d=0 is excluded for that packet.
        self.assertLess(estimator.probabilities[0], 1.0 / 6.0)

    def test_no_censoring_uses_only_prior_and_weighted_revealed_delays(self) -> None:
        estimator = CensoredAckDelayEstimator(
            max_delay=4,
            decay_time=5.0,
            prior_strength=0.5,
            prior_probabilities=np.ones(5),
            use_censoring=False,
        )
        pending = {2: [_ack(1, delay=0)], 5: [_ack(3, delay=1)]}
        for decision_step in range(6):
            probabilities = estimator.update(
                decision_step, pending.get(decision_step, ())
            )
        # Pending packets contribute nothing. Exact ACKs retain their weights.
        expected_counts = np.full(5, 0.1)
        expected_counts[0] += np.exp(-4.0 / 5.0)
        expected_counts[1] += np.exp(-2.0 / 5.0)
        np.testing.assert_allclose(probabilities, expected_counts / expected_counts.sum())

    def test_no_censoring_underestimates_delay_after_upward_changepoint(self) -> None:
        options = {
            "max_delay": 6,
            "memory_steps": 16,
            "decay_time": 5.0,
            "prior_strength": 0.2,
        }
        censored = CensoredAckDelayEstimator(**options)
        uncensored = CensoredAckDelayEstimator(**options, use_censoring=False)
        changepoint = 40
        support = np.arange(7)
        pending: dict[int, list[Ack]] = {}
        for generated_step in range(1, changepoint + 11):
            delay = 0 if generated_step < changepoint else 6
            ack = _ack(generated_step, delay=delay)
            pending.setdefault(ack.revealed_step, []).append(ack)

        for decision_step in range(changepoint + 6):
            revealed = pending.get(decision_step, ())
            censored_mean = float(censored.update(decision_step, revealed) @ support)
            uncensored_mean = float(uncensored.update(decision_step, revealed) @ support)
            if changepoint + 2 <= decision_step:
                # The first long-delay ACK is not yet revealed at these steps.
                with self.subTest(decision_step=decision_step):
                    self.assertLess(uncensored_mean, 0.25)
                    self.assertGreater(censored_mean - uncensored_mean, 0.75)

    def test_current_age_distribution_for_deterministic_delay(self) -> None:
        estimator = CensoredAckDelayEstimator(max_delay=4)
        estimator._probabilities = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0])
        np.testing.assert_allclose(
            estimator.effective_age_probabilities(10),
            np.asarray([0.0, 0.0, 1.0, 0.0, 0.0]),
        )
        np.testing.assert_allclose(
            estimator.effective_age_probabilities(1),
            np.asarray([0.0, 1.0]),
        )


if __name__ == "__main__":
    unittest.main()
