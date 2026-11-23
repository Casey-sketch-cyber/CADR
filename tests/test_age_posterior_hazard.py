from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest

from dt_corl_delay.channel import Ack, DelayedObservationChannel, PolicyInput
from dt_corl_delay.delay_belief import (
    CensoredAckDelayEstimator,
    ContentConditionedAgePosterior,
    hazard_age_probabilities,
    validate_age_posterior,
)
from dt_corl_delay.delay_process import CategoricalDelay


def test_hazard_normalizes_on_the_bounded_sparse_support() -> None:
    posterior = hazard_age_probabilities(np.array([0.1, 0.2, 0.0, 0.4, 0.3]), 3)
    assert posterior.shape == (4,)
    assert posterior.sum() == pytest.approx(1.0)
    assert posterior[2] == 0.0
    np.testing.assert_allclose(posterior, np.array([1, 2, 0, 4]) / 7)


@pytest.mark.parametrize("delay", [0, 2, 5])
def test_fixed_delay_is_a_point_mass_including_zero_survival(delay: int) -> None:
    q = np.zeros(6)
    q[delay] = 1.0
    np.testing.assert_array_equal(hazard_age_probabilities(q, 5), q)
    np.testing.assert_array_equal(hazard_age_probabilities(q, 0), [1.0])


def test_tiny_survival_mass_is_preserved() -> None:
    q = np.array([1.0, 1e-20, 0.0])
    np.testing.assert_allclose(hazard_age_probabilities(q, 2), q, atol=0, rtol=1e-15)


def test_hazard_rejects_unsupported_ack_lag_only_when_conditioning() -> None:
    with pytest.raises(ValueError, match="requires ack_lag=1"):
        validate_age_posterior("hazard", 2, True)
    validate_age_posterior("hazard", 2, False)
    validate_age_posterior("legacy_cdf", 2, True)
    with pytest.raises(ValueError, match="age_posterior"):
        validate_age_posterior("invalid", 1, True)


@pytest.mark.parametrize(
    ("delays", "probabilities"),
    [
        ((0, 1, 2), (0.65, 0.25, 0.10)),
        ((3, 5, 7), (0.20, 0.45, 0.35)),
        ((8, 12, 16), (0.20, 0.45, 0.35)),
    ],
)
def test_hazard_matches_true_q_channel_simulation(
    delays: tuple[int, ...], probabilities: tuple[float, ...]
) -> None:
    # Same experiment as scripts/revision/age_posterior_check.py. Unique
    # observations identify a change. The policy receives only revealed ACKs.
    q = np.zeros(delays[-1] + 1)
    q[list(delays)] = probabilities
    channel = DelayedObservationChannel(
        CategoricalDelay(delays, probabilities), random.Random(0), ack_lag=1
    )
    channel.reset(0, 0)
    previous_generation = 0
    acknowledged_generation = 0
    counts = np.zeros((q.size, q.size), dtype=np.int64)
    for step in range(1, 50_000):
        channel.send(step, step)
        outcome = channel.advance(step)
        assert outcome.selected_packet is not None
        true_generation = outcome.selected_packet.generated_step
        for ack in outcome.policy_input.acknowledgements:
            acknowledged_generation = max(acknowledged_generation, ack.generated_step)
        if true_generation != previous_generation:
            maximum_age = step - acknowledged_generation - 1
            counts[maximum_age, step - true_generation] += 1
        previous_generation = true_generation

    # Joint TV of (bound, age), weighting each bound by its empirical frequency.
    expected = np.zeros_like(counts, dtype=np.float64)
    counts_by_bound = counts.sum(axis=1)
    for maximum_age, count in enumerate(counts_by_bound):
        if count:
            posterior = hazard_age_probabilities(q, maximum_age)
            expected[maximum_age, : maximum_age + 1] = count * posterior
    total_variation = 0.5 * np.abs(expected - counts).sum() / counts.sum()
    assert total_variation < 0.02


def test_legacy_content_conditioning_is_bit_identical_to_frozen_v1() -> None:
    q = np.array([0.11, 0.13, 0.19, 0.22, 0.35])
    estimator = CensoredAckDelayEstimator(max_delay=4, prior_probabilities=q)
    # Set the frozen q directly to match the v1 golden fixture.
    estimator._probabilities = q
    legacy = ContentConditionedAgePosterior(4, "legacy_cdf")
    legacy.reset(np.array([0.0]))
    sequence = [
        (0.0, ()),
        (0.0, ()),
        (1.0, ()),
        (1.0, (Ack(1, 1, 2, 3, 1),)),
        (2.0, ()),
        (2.0, (Ack(3, 3, 4, 5, 1),)),
        (3.0, ()),
    ]
    # Captured from the unmodified v1 controller at 33d104e as exact float64.
    two_ages = ("0x1.5c158496a01e6p-2", "0x1.51f53db4aff0dp-1")
    three_ages = (
        "0x1.6ea2b90f9df48p-3",
        "0x1.63f8498dc42e8p-2",
        "0x1.e4b659ea6cd74p-2",
    )
    expected = [
        ("0x1p+0",),
        ("0x0p+0", "0x1p+0"),
        two_ages,
        ("0x0p+0", "0x0p+0", "0x1p+0"),
        three_ages,
        ("0x0p+0", "0x0p+0", "0x1p+0"),
        three_ages,
    ]
    for step, (observation, acks) in enumerate(sequence):
        posterior, exact, changed = legacy.probabilities(
            np.array([observation]),
            step,
            acks,
            estimator.effective_age_probabilities(step),
            q,
        )
        target = np.array([float.fromhex(value) for value in expected[step]])
        assert posterior.tobytes() == target.tobytes()
        assert exact == (step in {0, 1, 3, 5})
        assert changed == (step in {2, 4, 6})


def test_hazard_only_replaces_the_changed_observation_branch() -> None:
    q = np.array([0.40, 0.35, 0.25])
    estimator = CensoredAckDelayEstimator(max_delay=2, prior_probabilities=q)
    history = ContentConditionedAgePosterior(2, "hazard")
    initial = np.array([0.0])
    history.reset(initial)
    for step, expected in enumerate(([1.0], [0.0, 1.0])):
        posterior, exact, changed = history.probabilities(
            initial, step, (), estimator.effective_age_probabilities(step), q
        )
        np.testing.assert_array_equal(posterior, expected)
        assert exact and not changed
    posterior, exact, changed = history.probabilities(
        np.array([1.0]), 2, (), estimator.effective_age_probabilities(2), q
    )
    np.testing.assert_allclose(posterior, q[:2] / q[:2].sum())
    assert changed and not exact


@pytest.mark.parametrize("controller_kind", ["cadr", "dp"])
@pytest.mark.parametrize(("use_content", "expected_age"), [(True, 0.0), (False, 1.0)])
def test_hazard_reaches_the_controller_and_respects_no_content(
    controller_kind: str,
    use_content: bool,
    expected_age: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from dt_corl_delay.controllers import CadrController

    device = torch.device("cpu")
    common = {
        "agent": SimpleNamespace(
            device=device,
            act_normalized=lambda state: state,
            get_action=lambda observations: observations.new_zeros((1, 8, 1)),
        ),
        "dynamics": SimpleNamespace(
            rollout_age_candidates=lambda state, actions, maximum_age: (
                state.unsqueeze(0).expand(maximum_age + 1, -1, -1)
            ),
            state_dim=1,
            ensemble_size=1,
        ),
        "state_mean": torch.zeros(1, 1),
        "state_std": torch.ones(1, 1),
        "max_delay": 2,
        "ack_lag": 1,
        "age_posterior": "hazard",
        "use_content_conditioning": use_content,
    }
    if controller_kind == "dp":
        pytest.importorskip("h5py")
        from dt_corl_delay.maniskill_diffusion import CadrDiffusionPolicyController

        controller = CadrDiffusionPolicyController(device=device, **common)
    else:
        controller = CadrController(**common)
    controller.reset(np.array([0.0], dtype=np.float32))
    q = np.array([0.40, 0.35, 0.25])
    monkeypatch.setattr(controller.estimator, "_probabilities", q)
    monkeypatch.setattr(controller.estimator, "update", lambda step, acks: q)
    controller.act(PolicyInput(np.array([1.0], dtype=np.float32), ()), decision_step=2)
    assert controller.diagnostics()["mean_selected_age"] == expected_age


@pytest.mark.parametrize(
    ("settings", "expected_age"),
    [
        ({}, 1),
        ({"age_posterior": "legacy_cdf"}, 1),
        ({"age_posterior": "hazard"}, 0),
        ({"age_posterior": "hazard", "use_content_conditioning": False}, 1),
    ],
)
def test_dt_ack_opt_in_shares_hazard_and_preserves_v1_map(
    settings: dict[str, str | bool], expected_age: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    from dt_corl_delay.official_dtcorl import OfficialDTCorlAckMapController

    agent = SimpleNamespace(device=torch.device("cpu"), act_normalized=lambda state: state)
    autoencoder = SimpleNamespace(encode=lambda state: state, decode=lambda latent: latent)
    controller = OfficialDTCorlAckMapController(
        agent=agent,
        autoencoder=autoencoder,
        dynamics=lambda **inputs: inputs["latents"],
        state_mean=torch.zeros(1, 1),
        state_std=torch.ones(1, 1),
        maximum_delay=2,
        ack_lag=1,
        **settings,
    )
    controller.reset(np.array([0.0]))
    q = np.array([0.40, 0.35, 0.25])
    monkeypatch.setattr(controller.estimator, "_probabilities", q)
    monkeypatch.setattr(controller.estimator, "update", lambda step, acks: q)
    for _ in range(2):
        controller.observe_transition(np.zeros(1), 0.0, False, False)
    controller.act(PolicyInput(np.array([1.0]), ()), decision_step=2)
    assert controller.diagnostics()["mean_estimated_age"] == expected_age
