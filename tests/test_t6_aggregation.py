"""Check Hopper's aggregation and active ensemble on fixed inputs."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dt_corl_delay.channel import Ack, PolicyInput
from dt_corl_delay.controllers import CadrController, DirectObservationController
from dt_corl_delay.dynamics import DeltaDynamicsEnsemble, Float64AgeRolloutEnsemble


class RecordingActor:
    device = torch.device("cpu")

    def __init__(self):
        self.inputs = []

    def act_normalized(self, states):
        self.inputs.append(states.clone())
        return states.square() / 4

    def act(self, observation, state_mean, state_std):
        observation = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        return self.act_normalized((observation - state_mean) / state_std).squeeze(0).numpy()


def _dynamics(ensemble_class=DeltaDynamicsEnsemble):
    dynamics = ensemble_class(
        state_dim=1,
        action_dim=1,
        hidden_dims=(),
        ensemble_size=2,
        learning_rate=3e-4,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        for member, delta in zip(dynamics.members, (0.25, 0.75), strict=True):
            member[0].weight.zero_()
            member[0].bias.fill_(delta)
    return dynamics


def _observation(value):
    return np.array([value], dtype=np.float32)


def _controller(actor, dynamics, **overrides):
    options = {
        "agent": actor,
        "dynamics": dynamics,
        "state_mean": torch.tensor([[3.0]]),
        "state_std": torch.tensor([[2.0]]),
        "max_delay": 2,
        "ack_lag": 1,
        "age_posterior": "hazard",
        "posterior_updates": False,
        "fixed_prior_probabilities": [0.2, 0.6, 0.2],
    }
    controller = CadrController(**(options | overrides))
    controller.reset(_observation(3.0))
    return controller


def _actions(controller, inputs):
    return [controller.act(policy_input, step) for step, policy_input in enumerate(inputs)]


@pytest.mark.parametrize(
    ("delay_prior", "expected_action"),
    [
        pytest.param(None, None, id="online-ack-updates"),
        pytest.param([0.2, 0.6, 0.2], 0.265625, id="positive-map-age"),
        pytest.param([0.25, 0.25, 0.5], 0.0625, id="map-age-tie"),
    ],
)
def test_map_aggregation_is_bitwise_the_ack_map_hazard_rule(delay_prior, expected_action):
    reference_actor, alias_actor = RecordingActor(), RecordingActor()
    dynamics = _dynamics()
    delay_settings = {
        "posterior_updates": delay_prior is None,
        "fixed_prior_probabilities": delay_prior,
    }
    reference = _controller(
        reference_actor,
        dynamics,
        age_decision_rule="map",
        **delay_settings,
    )
    alias = _controller(
        alias_actor,
        dynamics,
        aggregation="map",
        **delay_settings,
    )
    inputs = [
        PolicyInput(_observation(3.0), ()),
        PolicyInput(_observation(3.0), ()),
        PolicyInput(_observation(4.0), ()),
        PolicyInput(_observation(5.0), (Ack(2, 2, 2, 3, 0),)),
    ]
    reference_actions = _actions(reference, inputs)
    alias_actions = _actions(alias, inputs)

    if expected_action is not None:
        assert reference_actions[2].item() == expected_action
    assert [value.tobytes() for value in alias_actions] == [
        value.tobytes() for value in reference_actions
    ]
    assert alias.diagnostics() == reference.diagnostics()
    assert len(alias_actor.inputs) == len(reference_actor.inputs)
    for actual, expected in zip(alias_actor.inputs, reference_actor.inputs, strict=True):
        assert actual.numpy().tobytes() == expected.numpy().tobytes()


def test_state_mean_weights_normalized_candidates_before_one_actor_forward():
    inputs = [
        PolicyInput(_observation(3.0), ()),
        PolicyInput(_observation(3.0), ()),
        PolicyInput(_observation(4.0), ()),
    ]
    dynamics = _dynamics()
    frozen_members = {
        name: value.clone() for name, value in dynamics.members.state_dict().items()
    }
    state_actor = RecordingActor()
    state_mean = _controller(state_actor, dynamics, aggregation="state_mean")
    action_mean = _controller(RecordingActor(), dynamics)
    explicit_default = _controller(RecordingActor(), dynamics, aggregation="action_mean")

    state_action = _actions(state_mean, inputs)[2]
    action_actions = _actions(action_mean, inputs)
    explicit_actions = _actions(explicit_default, inputs)

    # At step 2, p(0), p(1) = 1/4, 3/4 and the normalized candidate
    # states are (0.5, 0.5) for age 0 and (0.75, 1.25) for age 1.
    assert len(state_actor.inputs) == len(inputs)
    torch.testing.assert_close(state_actor.inputs[-1], torch.tensor([[0.875]]), rtol=0, atol=0)
    assert state_action.item() == 0.19140625
    assert action_actions[2].item() == 0.21484375
    assert [value.tobytes() for value in action_actions] == [
        value.tobytes() for value in explicit_actions
    ]
    assert "mean_action_disagreement" not in state_mean.diagnostics()
    assert state_mean.diagnostics()["mean_state_disagreement"] > 0.0
    for name, value in dynamics.members.state_dict().items():
        torch.testing.assert_close(value, frozen_members[name], rtol=0, atol=0)


@pytest.mark.parametrize("aggregation", ["action_mean", "state_mean", "map"])
def test_proven_zero_age_is_bitwise_direct_without_dynamics(aggregation):
    dynamics = _dynamics()
    for member in dynamics.members:
        member.register_forward_hook(lambda *_: pytest.fail("zero age called dynamics"))
    wrapped = _controller(RecordingActor(), dynamics, aggregation=aggregation)
    direct = DirectObservationController(RecordingActor(), wrapped.state_mean, wrapped.state_std)
    inputs = [
        PolicyInput(_observation(3.0), ()),
        PolicyInput(_observation(4.0), ()),
        PolicyInput(_observation(5.0), (Ack(1, 1, 1, 2, 0),)),
    ]
    assert [value.tobytes() for value in _actions(wrapped, inputs)] == [
        value.tobytes() for value in _actions(direct, inputs)
    ]


@pytest.mark.parametrize("ensemble_class", [DeltaDynamicsEnsemble, Float64AgeRolloutEnsemble])
def test_single_selected_member_is_the_only_member_run_and_weighted(ensemble_class):
    dynamics = _dynamics(ensemble_class)
    if ensemble_class is Float64AgeRolloutEnsemble:
        dynamics.double()
    member_calls = []
    for index, member in enumerate(dynamics.members):
        member.register_forward_hook(lambda *_, index=index: member_calls.append(index))
    actor = RecordingActor()
    controller = _controller(actor, dynamics, ensemble_members=[0])
    controller.act(PolicyInput(_observation(3.0), ()), 0)
    controller.act(PolicyInput(_observation(3.0), ()), 1)
    member_calls.clear()

    action = controller.act(PolicyInput(_observation(4.0), ()), 2)

    assert member_calls == [0]
    torch.testing.assert_close(actor.inputs[-1], torch.tensor([[0.5], [0.75]]), rtol=0, atol=0)
    assert action.item() == 0.12109375
    assert controller.diagnostics()["mean_epistemic_state_disagreement"] == 0.0


@pytest.mark.parametrize("indices", [[], [0, 0], [-1], [2], [0.0], [True], "0", 1])
def test_ensemble_member_selection_requires_distinct_valid_integer_indices(indices):
    with pytest.raises(ValueError, match="ensemble_members"):
        _controller(RecordingActor(), _dynamics(), ensemble_members=indices)
