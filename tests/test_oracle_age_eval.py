import json

import numpy as np
import pytest
import torch

from dt_corl_delay import delayed_eval
from dt_corl_delay.aggregate_formal_results import _aggregate, _load_rows
from dt_corl_delay.channel import PolicyInput
from dt_corl_delay.controllers import CadrController
from dt_corl_delay.maniskill_diffusion import CadrDiffusionPolicyController
from dt_corl_delay.oracle_age_eval import (
    CADR_ORACLE_AGE_METHOD,
    OracleAgeCadrController,
    include_cadr_oracle_age,
)


class _RecordingDynamics:
    ensemble_size = 1
    state_dim = 1

    def __init__(self):
        self.executed_suffixes = []

    def rollout_age_candidates(self, state, actions, maximum_age):
        suffix = [float(action.item()) for action in actions[-maximum_age:]]
        self.executed_suffixes.append(suffix)
        return state.unsqueeze(0) + torch.tensor(
            [0.0] + [sum(suffix[-age:]) for age in range(1, maximum_age + 1)]
        ).reshape(-1, 1, 1)


class _IqlActor:
    device = torch.device("cpu")

    def act_normalized(self, state):
        return state


class _FixedEpisode:
    _max_episode_steps = 4

    def reset(self, *, seed):
        return np.array([0.1], dtype=np.float32), {}

    def step(self, action):
        return np.array([0.2], dtype=np.float32), float(action.item()), False, False, {}

    def close(self):
        pass


def test_privileged_evaluator_uses_observation_age_and_executes_only_past_actions(monkeypatch):
    dynamics = _RecordingDynamics()
    oracle = OracleAgeCadrController(
        agent=_IqlActor(),
        dynamics=dynamics,
        state_mean=torch.zeros((1, 1)),
        state_std=torch.ones((1, 1)),
        max_delay=1,
        ack_lag=1,
    )
    assert not hasattr(CadrController, "set_oracle_age")
    monkeypatch.setattr(delayed_eval, "_recover_environment", lambda *_: _FixedEpisode())
    result = delayed_eval.evaluate_controller(
        controller_factory=lambda: oracle,
        dataset_id="unused",
        scenario={"name": "fixed_2", "type": "fixed", "delay": 2},
        episodes=1,
        seed=10,
        ack_lag=1,
        cvar_alpha=0.2,
        method=CADR_ORACLE_AGE_METHOD,
        privileged_selected_age=True,
    )
    # The bootstrap packet has sampled delay zero. Its held ages are 1 and
    # then 2, even though the deployable estimator support was set to 1.
    assert [len(suffix) for suffix in dynamics.executed_suffixes] == [1, 2, 2]
    assert dynamics.executed_suffixes[0] == pytest.approx([0.1])
    assert dynamics.executed_suffixes[1] == pytest.approx([0.1, 0.2])
    assert dynamics.executed_suffixes[2] == pytest.approx([0.2, 0.4])
    assert result["returns"] == pytest.approx([1.5])
    assert result["mean_selected_age"] == 1.25
    assert result["privileged_selected_age"] is True
    assert result["method"] == CADR_ORACLE_AGE_METHOD
    with pytest.raises(RuntimeError, match="supply oracle age"):
        oracle.act(PolicyInput(np.zeros(1), ()), decision_step=4)


class _DiffusionActor:
    def __init__(self):
        self.observation_sequences = []

    def get_action(self, observation_sequence):
        self.observation_sequences.append(observation_sequence.clone())
        return torch.tensor([[[0.1], [0.2]]])


def test_privileged_diffusion_uses_true_age_without_changing_chunk_schedule():
    agent = _DiffusionActor()
    dynamics = _RecordingDynamics()
    oracle = CadrDiffusionPolicyController(
        agent=agent,
        dynamics=dynamics,
        state_mean=torch.zeros((1, 1)),
        state_std=torch.ones((1, 1)),
        device=torch.device("cpu"),
        max_delay=1,
        ack_lag=1,
        obs_horizon=2,
        act_horizon=2,
        age_source=CADR_ORACLE_AGE_METHOD,
    )
    observation = np.zeros(1, dtype=np.float32)
    oracle.reset(observation)
    actions = []
    for step in range(3):
        oracle.set_oracle_age(step)
        action = oracle.act(PolicyInput(observation, ()), decision_step=step)
        actions.append(float(action.item()))
    assert actions == pytest.approx([0.1, 0.2, 0.1])
    assert len(agent.observation_sequences) == 2
    assert agent.observation_sequences[1].flatten().tolist() == pytest.approx([0.1, 0.3])
    oracle.set_oracle_age(2)
    oracle.reset(observation)
    with pytest.raises(RuntimeError, match="before every action"):
        oracle.act(PolicyInput(observation, ()), decision_step=0)


def test_v2_tables_identify_privileged_age_and_v1_keeps_its_schema(tmp_path):
    summary = {
        "evaluations": {
            "fixed_2": {
                "oracle": {
                    "method": CADR_ORACLE_AGE_METHOD,
                    "privileged_selected_age": True,
                    "returns": [10],
                    "episode_seeds": [20],
                }
            }
        }
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    config_path = tmp_path / "config.json"
    for version in ("v1", "v2"):
        series = f"formal_{version}_synthetic"
        config_path.write_text(json.dumps({"experiment_series": series}), encoding="utf-8")
        rows = _load_rows([tmp_path], series)
        aggregates, _ = _aggregate(rows, bootstrap_seed=1, resamples=5)
        for record in (rows[0], aggregates[0]):
            assert record["method"] == CADR_ORACLE_AGE_METHOD
            if version == "v2":
                assert record["privileged_selected_age"] is True
            else:
                assert "privileged_selected_age" not in record


def test_oracle_is_opt_in_and_rejected_on_v1():
    assert not include_cadr_oracle_age({"experiment_series": "formal_v1_minari_synthetic"})
    with pytest.raises(ValueError, match="v2"):
        include_cadr_oracle_age({
            "experiment_series": "formal_v1_minari_synthetic",
            "include_cadr_oracle_age": True,
        })
    assert include_cadr_oracle_age({
        "experiment_series": "formal_v2_minari_synthetic",
        "include_cadr_oracle_age": True,
    })
