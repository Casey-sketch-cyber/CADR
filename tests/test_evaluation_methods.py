import json
from types import SimpleNamespace

import pytest
import torch

from dt_corl_delay import evaluate_checkpoints as iql_eval
from dt_corl_delay import evaluate_maniskill_cadr_diffusion as dp_eval
from dt_corl_delay.evaluation_methods import select_evaluation_methods


def test_select_evaluation_methods_preserves_default_order_and_checks_aliases():
    available = ("direct", "cadr_dp_fixed_prior")
    aliases = {"cadr": "cadr_dp_fixed_prior"}
    assert select_evaluation_methods({}, available, aliases) == available
    assert select_evaluation_methods(
        {"evaluation_methods": ["cadr", "direct"]}, available, aliases
    ) == available
    for names in ([], "direct", [1], ["oracle_dp"], ["cadr", "cadr_dp_fixed_prior"]):
        with pytest.raises(ValueError, match="evaluation_method|evaluation method"):
            select_evaluation_methods({"evaluation_methods": names}, available, aliases)


class _Dynamics:
    def __init__(self, **kwargs):
        self.ensemble_size = kwargs["ensemble_size"]

    def load_checkpoint(self, _):
        pass

    def eval(self):
        pass


def _stub_method_calls(monkeypatch, module):
    calls = []

    def evaluate_controller(**kwargs):
        calls.append((kwargs["method"], kwargs.get("privileged_selected_age", False)))
        return {
            "method": kwargs["method"],
            "mean_return": 2.0,
            "returns": [2.0],
            "episode_lengths": [1],
            "successes": [1],
            "successes_at_end": [1],
            "success_occupancies": [1.0],
        }

    monkeypatch.setattr(module, "evaluate_controller", evaluate_controller)
    monkeypatch.setattr(module, "DeltaDynamicsEnsemble", _Dynamics)
    return calls


def _input_files(tmp_path):
    names = ("policy.pt", "dynamics.pt", "dataset.h5")
    for name in names:
        (tmp_path / name).write_bytes(b"test evaluation input")
    return tuple(tmp_path / name for name in names)


def _base_config():
    return {
        "seed": 101,
        "eval_seed": 12,
        "eval_episodes": 1,
        "dataset_id": "fixture",
        "device": "cpu",
        "ack_lag": 1,
        "cvar_alpha": 0.2,
        "max_delay": 8,
        "posterior_memory_steps": 32,
        "posterior_decay_time": 16.0,
        "posterior_prior_strength": 1.0,
        "scenarios": [{"name": "clean_delay_0", "type": "fixed", "delay": 0}],
    }


def _dynamics_payload():
    return {
        "config": {
            "dataset_id": "fixture",
            "dynamics_hidden_dims": [1],
            "dynamics_learning_rate": 0.01,
            "ensemble_size": 1,
        },
        "dynamics": {"state_dim": 1, "action_dim": 1, "ensemble_size": 1},
        "step": 1,
    }


def test_iql_evaluates_only_requested_method_and_keeps_legacy_default(monkeypatch, tmp_path):
    policy, dynamics, _ = _input_files(tmp_path)
    calls = _stub_method_calls(monkeypatch, iql_eval)
    payload = {
        "dataset_metadata": {"dataset_id": "fixture"},
        "config": {"algorithm": "iql"},
        "state_mean": torch.zeros(1),
        "state_std": torch.ones(1),
    }
    monkeypatch.setattr(iql_eval, "_load_agent", lambda *args: object())
    monkeypatch.setattr(iql_eval.torch, "load", lambda path, **_: (
        payload if path == policy else _dynamics_payload()
    ))
    config_path = tmp_path / "config.json"
    config = _base_config()
    config_path.write_text(json.dumps(config), encoding="utf-8")
    legacy = iql_eval.run(config_path, policy, dynamics, tmp_path / "legacy")
    assert calls == [("direct_observation_iql", False), ("cadr_iql", False)]
    assert legacy["evaluations"]["clean_delay_0"]["paired_mean_return_delta"] == 0
    assert "evaluation_methods" not in legacy

    calls.clear()
    config.update(evaluation_methods=["cadr_oracle_age"], include_cadr_oracle_age=True,
                  experiment_series="formal_v2_t6_hopper")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    oracle = iql_eval.run(config_path, policy, dynamics, tmp_path / "oracle")
    assert calls == [("cadr_oracle_age", True)]
    assert set(oracle["evaluations"]["clean_delay_0"]) == {"cadr_oracle_age", "wall_seconds"}


def test_dp_selected_variant_keeps_result_keys_and_can_resume(monkeypatch, tmp_path):
    policy, dynamics, dataset = _input_files(tmp_path)
    calls = _stub_method_calls(monkeypatch, dp_eval)
    monkeypatch.setattr(dp_eval, "official_maniskill_source_provenance", lambda *args: {})
    monkeypatch.setattr(dp_eval, "load_official_diffusion_agent", lambda **_: (object(), {}))
    monkeypatch.setattr(dp_eval, "load_offline_transitions", lambda *args: SimpleNamespace(
        metadata={"state_dim": 1, "action_dim": 1},
        state_mean=torch.zeros(1),
        state_std=torch.ones(1),
    ))
    monkeypatch.setattr(dp_eval.torch, "load", lambda *args, **kwargs: _dynamics_payload())
    config = _base_config() | {
        "dataset_path": str(dataset),
        "environment": {},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    legacy = dp_eval.run(config_path, policy, dynamics, tmp_path, tmp_path / "legacy")
    assert calls == [
        ("official_maniskill_dp_direct_delayed_observation", False),
        ("cadr_posterior_state_reconstruction_plus_official_maniskill_dp", False),
    ]
    assert legacy["paired_clean_equivalence"]["all_required_fields_exactly_equal"]

    calls.clear()
    config.update(evaluation_methods=["cadr"], ablation_variant="no_censoring")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "selected"
    selected = dp_eval.run(config_path, policy, dynamics, tmp_path, output)
    assert calls == [("cadr_dp_ablation_no_censoring_plus_official_maniskill_dp", False)]
    assert set(selected["evaluations"]["clean_delay_0"]) == {"cadr_dp_no_censoring"}
    assert selected["paired_clean_equivalence"] is None
    # A method persisted to progress.json must not be rerun even if the final
    # summary has not been written, as after a process interruption.
    (output / "summary.json").unlink()
    calls.clear()
    resumed = dp_eval.run(config_path, policy, dynamics, tmp_path, output, resume=True)
    assert not calls
    assert resumed["evaluations"] == selected["evaluations"]
