"""Verify T10 age decisions and the native IQL and DP policy paths."""

import numpy as np
import pytest
import torch

from dt_corl_delay.channel import Ack, PolicyInput
from dt_corl_delay.controllers import CadrController
from dt_corl_delay.maniskill_diffusion import CadrDiffusionPolicyController
from dt_corl_delay.upper_quantile import upper_age_quantile, upper_quantile_method


class NonlinearPolicy:
    device = torch.device("cpu")

    def act_normalized(self, state):
        return state.square() / 16

    def get_action(self, observations):
        return observations[:, -1:, :].square().expand(-1, 2, -1) / 16


class MemberDynamics:
    state_dim = 1
    ensemble_size = 2

    def __init__(self):
        self.horizons = []

    def rollout_age_candidates(self, observation, actions, maximum_age):
        assert maximum_age <= len(actions)
        self.horizons.append(maximum_age)
        ages = torch.arange(maximum_age + 1, dtype=observation.dtype)[:, None, None]
        return observation + ages * observation.new_tensor([1.0, 3.0])[None, :, None]


def controller(kind, **options):
    parameters = {
        "agent": NonlinearPolicy(), "dynamics": MemberDynamics(),
        "state_mean": torch.zeros(1), "state_std": torch.ones(1), "max_delay": 3, "ack_lag": 1,
    } | options
    if kind == "dp":
        return CadrDiffusionPolicyController(**parameters, device=torch.device("cpu"), act_horizon=2)
    return CadrController(**parameters)


def run_steps(model):
    packets = [(0.0, ()), (0.0, ()), (1.0, ()), (1.0, (Ack(1, 1, 2, 3, 1),)),
               (2.0, (Ack(2, 2, 3, 4, 1),))]
    initial = np.array([0.0], dtype=np.float32)
    model.reset(initial)
    actions = [model.act(PolicyInput(np.array([value], dtype=np.float32), acks), step)
               for step, (value, acks) in enumerate(packets)]
    return np.stack(actions), model._diagnostics["selected_age"]


def test_quantile_cdf_ties_and_single_feasible_age():
    posterior = np.array([0.25, 0.0, 0.25, 0.5], dtype=np.float64)
    assert upper_age_quantile(posterior, 0.25) == 0
    assert upper_age_quantile(posterior, np.nextafter(0.25, 1.0)) == 2
    assert upper_age_quantile(posterior, 0.5) == 2
    assert upper_age_quantile(np.array([1.0]), 0.90) == 0
    assert upper_age_quantile(np.array([1.0]), 1.0) == 0
    # The feasible tail remains eligible when the preceding CDF rounds to one.
    assert upper_age_quantile(np.array([1.0, 1e-18]), 1.0) == 1


@pytest.mark.parametrize("alpha", [None, True, "0.90", 0, -0.1, 1.01, np.inf, np.nan])
def test_upper_quantile_requires_finite_numeric_alpha(alpha):
    with pytest.raises(ValueError, match="finite numeric age_quantile"):
        upper_quantile_method({"age_decision_rule": "upper_quantile", "age_quantile": alpha})


def test_upper_quantile_name_is_opt_in():
    assert upper_quantile_method({}) is None
    assert upper_quantile_method({"age_decision_rule": "map", "age_quantile": 0.90}) is None
    assert upper_quantile_method({"age_decision_rule": "upper_quantile", "age_quantile": 0.90}) == "upper_quantile_a90"
    assert upper_quantile_method({"age_decision_rule": "upper_quantile", "age_quantile": 1}) == "upper_quantile_a100"


@pytest.mark.parametrize("kind", ["iql", "dp"])
def test_alpha_one_matches_in_order_actions_bitwise(kind):
    quantile = controller(kind, age_decision_rule="upper_quantile", age_quantile=1.0, age_posterior="hazard")
    reference = controller(kind, age_decision_rule="in_order_age")
    actions, ages = run_steps(quantile)
    reference_actions, reference_ages = run_steps(reference)
    assert actions.tobytes() == reference_actions.tobytes()
    assert ages == reference_ages == [0, 1, 1, 2, 1]
    assert quantile.dynamics.horizons == reference.dynamics.horizons
    # IQL averages nonlinear actor outputs for the two selected-age states,
    # while DP uses their average state in its ordinary single chunk inference.
    assert actions[2, 0] == {"iql": 0.625, "dp": 0.5625}[kind]


@pytest.mark.parametrize("kind", ["iql", "dp"])
def test_upper_quantile_preserves_unchanged_age_and_zero_age_bypass(kind):
    model = controller(kind, age_decision_rule="upper_quantile", age_quantile=0.01, age_posterior="hazard")
    actions, ages = run_steps(model)
    assert ages == [0, 1, 0, 2, 0]
    assert model.dynamics.horizons == [1, 2]
    assert actions[2, 0] == 0.0625
    assert model._diagnostics["exact_age_from_unchanged_observation"] == [1, 1, 0, 1, 0]


@pytest.mark.parametrize("kind", ["iql", "dp"])
def test_only_action_mean_and_content_conditioned_hazard_are_allowed(kind):
    settings = {"age_decision_rule": "upper_quantile", "age_quantile": 0.9, "age_posterior": "hazard"}
    with pytest.raises(ValueError, match="action_mean"):
        controller(kind, **(settings | {"aggregation": "state_mean"}))
    with pytest.raises(ValueError, match="content-conditioned hazard"):
        controller(kind, **(settings | {"use_content_conditioning": False}))
    with pytest.raises(ValueError, match="content-conditioned hazard"):
        controller(kind, **(settings | {"age_posterior": "legacy_cdf"}))
    with pytest.raises(ValueError, match="positive support"):
        controller(kind, **(settings | {"fixed_prior_probabilities": [1, 0, 0, 0], "posterior_updates": False}))


@pytest.mark.parametrize("kind", ["iql", "dp"])
def test_legacy_default_remains_bitwise_explicit_posterior_mean(kind):
    implicit, implicit_ages = run_steps(controller(kind))
    explicit, explicit_ages = run_steps(controller(kind, age_decision_rule="posterior_mean", age_posterior="legacy_cdf"))
    assert implicit.tobytes() == explicit.tobytes()
    assert implicit_ages == explicit_ages
