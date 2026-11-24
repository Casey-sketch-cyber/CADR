import unittest

import numpy as np
import torch

from dt_corl_delay.channel import Ack, PolicyInput
from dt_corl_delay.controllers import CadrController
from dt_corl_delay.maniskill_diffusion import CadrDiffusionPolicyController
from dt_corl_delay.simple_baselines import resolve_simple_baseline


class _Policy:
    device = torch.device("cpu")

    def act_normalized(self, state):
        return torch.zeros_like(state)

    def get_action(self, observations):
        return observations.new_zeros((1, 2, 1))


class _Dynamics:
    state_dim = 1
    ensemble_size = 1

    def __init__(self):
        self.age_horizons = []

    def rollout_age_candidates(self, observation, actions, maximum_age):
        if maximum_age > len(actions):
            raise AssertionError("rollout may use only executed actions")
        self.age_horizons.append(maximum_age)
        return observation.expand(maximum_age + 1, 1, 1)


class _ObservationWithoutAcks:
    observation = np.asarray([0.0], dtype=np.float32)

    @property
    def acknowledgements(self):
        raise AssertionError("fixed-horizon controller accessed ACKs")


class SimpleBaselineTest(unittest.TestCase):
    def _construct(self, controller_name, **options):
        parameters = {
            "agent": _Policy(),
            "dynamics": _Dynamics(),
            "state_mean": torch.zeros(1),
            "state_std": torch.ones(1),
            "max_delay": 8,
            "ack_lag": 1,
        }
        parameters.update(options)
        if controller_name == "dp":
            return CadrDiffusionPolicyController(
                **parameters, device=torch.device("cpu"), act_horizon=2
            )
        return CadrController(**parameters)

    def test_in_order_uses_content_changes_and_revealed_generation_steps(self):
        known = Ack(1, 1, 2, 3, 1)
        packet_values = [0.0, 0.0, 1.0, 1.0, 2.0]
        for kind in ("iql", "dp"):
            with self.subTest(kind=kind):
                controller = self._construct(kind, age_decision_rule="in_order_age")
                controller.reset(np.asarray([0.0], dtype=np.float32))
                for step, value in enumerate(packet_values):
                    observation = np.asarray([value], dtype=np.float32)
                    acks = (known,) if step == 3 else ()
                    controller.act(PolicyInput(observation, acks), step)
                self.assertIsNone(controller.estimator)
                self.assertEqual(controller.dynamics.age_horizons, [1, 1, 2, 2])
                diagnostics = controller.diagnostics()
                self.assertEqual(diagnostics["mean_selected_age"], 1.2)
                self.assertEqual(
                    diagnostics["mean_exact_age_from_unchanged_observation"], 0.6
                )
                self.assertNotIn("mean_posterior_delay_mean", diagnostics)

    def test_in_order_rejects_unrevealed_acks_and_non_unit_lag(self):
        with self.assertRaisesRegex(ValueError, "requires ack_lag=1"):
            self._construct("iql", age_decision_rule="in_order_age", ack_lag=2)
        controller = self._construct("iql", age_decision_rule="in_order_age")
        observation = np.asarray([0.0], dtype=np.float32)
        controller.reset(observation)
        ack = Ack(1, 1, 2, 3, 1)
        with self.assertRaisesRegex(ValueError, "before its reveal step"):
            controller.act(PolicyInput(observation, (ack,)), 2)

    def test_fixed_horizon_ignores_acks_and_only_rolls_through_available_actions(self):
        for kind in ("iql", "dp"):
            with self.subTest(kind=kind):
                controller = self._construct(
                    kind, age_decision_rule="fixed_horizon_model", fixed_horizon=2
                )
                policy_input = _ObservationWithoutAcks()
                controller.reset(policy_input.observation)
                for step in range(4):
                    controller.act(policy_input, step)
                self.assertIsNone(controller.estimator)
                self.assertEqual(controller.dynamics.age_horizons, [1, 2, 2])
                self.assertNotIn("mean_observation_changed", controller.diagnostics())
                controller.reset(policy_input.observation)
                controller.act(policy_input, 0)
                self.assertEqual(controller.diagnostics()["mean_selected_age"], 0.0)

    def test_fixed_horizon_uses_one_nominal_reference_in_all_scenarios(self):
        config = {
            "age_decision_rule": "fixed_horizon_model",
            "max_delay": 8,
            "scenarios": [
                {"name": "clean", "type": "fixed", "delay": 0},
                {
                    "name": "change",
                    "type": "changepoint",
                    "reference_delay_distribution": {
                        "delays": [1, 3, 8],
                        "probabilities": [0.75, 0.25, 0.0],
                    },
                },
                {"name": "network", "type": "trace"},
            ],
        }
        self.assertEqual(
            resolve_simple_baseline(config),
            ("fixed_horizon_model_mean", {"clean": 1, "change": 1, "network": 1}),
        )
        self.assertEqual(
            resolve_simple_baseline(config | {"fixed_horizon_rule": "max_support"}),
            ("fixed_horizon_model_max_support", {"clean": 2, "change": 2, "network": 2}),
        )
        with self.assertRaisesRegex(ValueError, "exceeds max_delay"):
            resolve_simple_baseline(config | {"max_delay": 0})
