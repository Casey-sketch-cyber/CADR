import json

import numpy as np
import pytest
import torch

from dt_corl_delay import delayed_eval, evaluate_checkpoints
from dt_corl_delay.channel import PolicyInput
from dt_corl_delay.controllers import CadrController
from dt_corl_delay.oracle_age_eval import (
    CADR_ORACLE_AGE_METHOD,
    CADR_ORACLE_AGE_OFFSET_METHOD,
    OracleAgeCadrController,
    OracleAgeOffsetCadrController,
    configured_oracle_age_offset,
)


def offset_config(offset):
    return {
        "experiment_series": "ablation_v2_p2b1_hopper",
        "evaluation_methods": [CADR_ORACLE_AGE_OFFSET_METHOD],
        "oracle_age_offset": offset,
    }


@pytest.mark.parametrize("offset", [-2, -1, 0, 1, 2])
def test_offset_valid_values_include_zero_as_an_explicit_opt_in(offset):
    assert configured_oracle_age_offset(offset_config(offset)) == offset
    assert configured_oracle_age_offset({}) is None


@pytest.mark.parametrize("offset", [False, True, -3, 3, 1.0, "1", None])
def test_offset_rejects_invalid_values(offset):
    with pytest.raises(ValueError, match="must be an integer"):
        configured_oracle_age_offset(offset_config(offset))


@pytest.mark.parametrize("bad_config", [
    {"experiment_series": "formal_v2_t6_hopper"},
    {"experiment_series": "ablation_v2_p2b1_"},
    {"evaluation_methods": ["direct", CADR_ORACLE_AGE_OFFSET_METHOD]},
    {"evaluation_methods": [CADR_ORACLE_AGE_METHOD]},
    {"include_cadr_oracle_age": True},
    {"include_oracle_upper_bound": True},
])
def test_offset_requires_its_own_series_and_method(bad_config):
    with pytest.raises(ValueError, match="oracle_age_offset requires|separate config"):
        configured_oracle_age_offset(offset_config(0) | bad_config)


class Actor:
    device = torch.device("cpu")

    def act_normalized(self, state):
        return state


class Dynamics:
    state_dim = ensemble_size = 1

    def __init__(self, **_):
        pass

    def load_checkpoint(self, _):
        pass

    def eval(self):
        pass

    def rollout_age_candidates(self, observation, actions, maximum_age):
        assert maximum_age <= len(actions)
        return observation + torch.arange(maximum_age + 1).reshape(-1, 1, 1) / 8


def make_oracle(offset=None, *, max_delay=4):
    parameters = dict(agent=Actor(), dynamics=Dynamics(), state_mean=torch.zeros(1),
                      state_std=torch.ones(1), max_delay=max_delay, ack_lag=1)
    if offset is None:
        return OracleAgeCadrController(**parameters)
    return OracleAgeOffsetCadrController(**parameters, oracle_age_offset=offset)


@pytest.mark.parametrize(("offset", "selected", "counts"), [
    (2, [0, 1, 2, 2, 2], (5, 4, 2)),
    (-2, [0, 0, 0, 0, 0], (5, -3, 3)),
])
def test_offset_clips_at_zero_current_step_and_max_delay_and_resets(offset, selected, counts):
    oracle = make_oracle(offset, max_delay=2)
    observation = np.asarray([0.125], dtype=np.float32)
    oracle.reset(observation)
    for step, age in enumerate([0, 0, 1, 2, 0]):
        oracle.set_oracle_age(age)
        action = oracle.act(PolicyInput(observation, ()), decision_step=step)
        assert action[0] == (1 + selected[step]) / 8
    assert oracle.age_offset_counts() == counts

    oracle.set_oracle_age(1)
    oracle.reset(observation)
    assert oracle.age_offset_counts() == (0, 0, 0)
    with pytest.raises(RuntimeError, match="supply oracle age"):
        oracle.act(PolicyInput(observation, ()), decision_step=0)


class Episodes:
    _max_episode_steps = 5

    def __init__(self):
        self.actions = []

    def reset(self, *, seed):
        self.limit = 2 if seed == 10 else 5
        self.step_number = 0
        return np.array([0.125], dtype=np.float32), {}

    def step(self, action):
        self.step_number += 1
        self.actions.append(action.copy())
        return np.array([0.25], dtype=np.float32), float(action.item()), self.step_number == self.limit, False, {}

    def close(self):
        pass


def evaluate(oracle, method, **settings):
    return delayed_eval.evaluate_controller(
        controller_factory=lambda: oracle,
        dataset_id="fixture",
        scenario={"name": "fixed_2", "type": "fixed", "delay": 2},
        episodes=2,
        seed=10,
        ack_lag=1,
        cvar_alpha=0.2,
        method=method,
        **({"privileged_selected_age": True} | settings),
    )


def test_evaluator_forwards_true_age_and_step_weights_diagnostics(monkeypatch):
    environment = Episodes()
    monkeypatch.setattr(delayed_eval, "_recover_environment", lambda *_: environment)
    oracle = make_oracle(1)
    true_ages = []
    original_setter = oracle.set_oracle_age

    def track_true_age(age):
        true_ages.append(age)
        original_setter(age)

    monkeypatch.setattr(oracle, "set_oracle_age", track_true_age)
    result = evaluate(oracle, CADR_ORACLE_AGE_OFFSET_METHOD)
    assert result["privileged_selected_age"] is True
    assert true_ages == [0, 1, 0, 1, 2, 2, 2]
    assert result["episode_lengths"] == [2, 5]
    assert result["age_offset_diagnostics"] == {
        "decision_steps": 7, "effective_offset_sum": 2, "fully_clipped_steps": 5,
        "mean_applied_offset": 2 / 7, "fully_clipped_fraction": 5 / 7,
    }


def test_zero_matches_original_oracle_bitwise_when_true_age_is_in_support(monkeypatch):
    old_environment, zero_environment = Episodes(), Episodes()
    environments = iter([old_environment, zero_environment])
    monkeypatch.setattr(delayed_eval, "_recover_environment", lambda *_: next(environments))
    previous = evaluate(make_oracle(), CADR_ORACLE_AGE_METHOD)
    shifted = evaluate(make_oracle(0), CADR_ORACLE_AGE_OFFSET_METHOD)
    assert np.stack(old_environment.actions).tobytes() == np.stack(zero_environment.actions).tobytes()
    assert np.asarray(previous["returns"], dtype=np.float64).tobytes() == np.asarray(
        shifted["returns"], dtype=np.float64,
    ).tobytes()
    assert previous["episode_seeds"] == shifted["episode_seeds"]
    assert previous["episode_lengths"] == shifted["episode_lengths"]
    assert "age_offset_diagnostics" not in previous
    assert shifted["age_offset_diagnostics"] == {
        "decision_steps": 7, "effective_offset_sum": 0, "fully_clipped_steps": 0,
        "mean_applied_offset": 0.0, "fully_clipped_fraction": 0.0,
    }
    assert not hasattr(CadrController, "set_oracle_age")


def test_generic_evaluator_rejects_offset_without_the_privileged_label_and_controller(monkeypatch):
    monkeypatch.setattr(delayed_eval, "_recover_environment", lambda *_: Episodes())
    with pytest.raises(ValueError, match="requires privileged selected age"):
        evaluate(make_oracle(1), CADR_ORACLE_AGE_OFFSET_METHOD, privileged_selected_age=False)
    with pytest.raises(ValueError, match="dedicated privileged controller"):
        evaluate(make_oracle(1), CADR_ORACLE_AGE_METHOD)
    with pytest.raises(ValueError, match="dedicated privileged controller"):
        evaluate(make_oracle(), CADR_ORACLE_AGE_OFFSET_METHOD)


def test_checkpoint_entrypoint_runs_only_selected_offset_method(monkeypatch, tmp_path):
    policy_path, dynamics_path = tmp_path / "p.pt", tmp_path / "d.pt"
    policy_path.write_bytes(b"policy fixture")
    dynamics_path.write_bytes(b"dynamics fixture")
    config = offset_config(0) | {
        "seed": 101, "eval_seed": 10, "eval_episodes": 2, "dataset_id": "fixture", "device": "cpu",
        "max_delay": 4, "ack_lag": 1, "cvar_alpha": 0.2, "posterior_memory_steps": 32,
        "posterior_decay_time": 16.0, "posterior_prior_strength": 1.0,
        "scenarios": [{"name": "fixed_2", "type": "fixed", "delay": 2}],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(evaluate_checkpoints, "_load_agent", lambda *_: Actor())
    monkeypatch.setattr(evaluate_checkpoints, "DeltaDynamicsEnsemble", Dynamics)
    monkeypatch.setattr(delayed_eval, "_recover_environment", lambda *_: Episodes())
    checkpoint_payloads = {
        policy_path: {"dataset_metadata": {"dataset_id": "fixture"}, "config": {"algorithm": "iql"},
                      "state_mean": torch.zeros(1), "state_std": torch.ones(1)},
        dynamics_path: {"config": {"dynamics_hidden_dims": [], "dynamics_learning_rate": 0.01},
                        "dynamics": {"state_dim": 1, "action_dim": 1, "ensemble_size": 1}},
    }
    monkeypatch.setattr(evaluate_checkpoints.torch, "load", lambda path, **_: checkpoint_payloads[path])
    summary = evaluate_checkpoints.run(config_path, policy_path, dynamics_path, tmp_path / "offset")
    assert summary["oracle_age_offset"] == 0
    assert summary["evaluation_methods"] == [CADR_ORACLE_AGE_OFFSET_METHOD]
    assert summary["included_privileged_cadr_oracle_age_offset"] is True
    assert summary["information_boundary"] == "privileged_selected_age_evaluation_only"
    scenario = summary["evaluations"]["fixed_2"]
    assert set(scenario) == {CADR_ORACLE_AGE_OFFSET_METHOD, "wall_seconds"}
    method = scenario[CADR_ORACLE_AGE_OFFSET_METHOD]
    assert method["method"] == CADR_ORACLE_AGE_OFFSET_METHOD
    assert method["privileged_selected_age"] is True
    assert method["age_offset_diagnostics"]["decision_steps"] == 7
    with pytest.raises(ValueError, match="own frozen CADR evaluation config"):
        evaluate_checkpoints.run(config_path, policy_path, dynamics_path, tmp_path / "direct", include_cadr=False)
