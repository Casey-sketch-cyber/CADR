import csv
import importlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/revision"))
    return importlib.import_module("aggregate_p2b1")


def result(module, scene, method, expected_seeds, value, length, offset):
    steps = len(expected_seeds) * length
    clipped = steps // 2 if offset and length == 2 else 0
    total = offset * (steps - clipped)
    return {
        "scenario": scene,
        "method": method,
        "privileged_selected_age": True,
        "episodes": len(expected_seeds),
        "episode_seeds": expected_seeds,
        "episode_lengths": [length] * len(expected_seeds),
        "returns": [value] * len(expected_seeds),
        "age_offset_diagnostics": {
            "decision_steps": steps,
            "effective_offset_sum": total,
            "fully_clipped_steps": clipped,
            "mean_applied_offset": total / steps,
            "fully_clipped_fraction": clipped / steps,
        },
    }


def put(module, path, value):
    module.write_json(path, value)
    return module.load_json(path)[1]


@pytest.fixture
def staged(module, tmp_path, monkeypatch):
    references = {}
    runtime = {
        "device": "Tesla T4", "cuda": "12.8", "torch": "2.8.0+cu128",
        "dynamics_rollout_dtype": "float32",
    }
    gpu = {"name": "Tesla T4", "memory.total": "15360", "driver_version": "580.82.07"}
    packages = {"torch": "2.8.0+cu128"}
    frozen_scenes = [{"name": scene} for scene in module.SCENARIOS]
    manifest = {
        "schema_version": 1,
        "stage": "p2b1",
        "p2a_plan_sha256": module.P2A_PLAN_SHA256,
        "decision_scenarios": list(module.DECISION_SCENARIOS),
        "descriptive_scenarios": list(module.DESCRIPTIVE_SCENARIOS),
        "eval_episodes": 20,
        "new_episodes": 800,
        "reference_episodes": 200,
        "preflight_episodes": 20,
        "runs": [],
    }
    manifest_path = tmp_path / "manifest.json"
    outputs = tmp_path / "runs"
    output = tmp_path / "aggregate"
    digests = {}
    for seed in module.SEEDS:
        config = {
            "seed": seed, "eval_seed": 31000001, "eval_episodes": 20,
            "scenarios": frozen_scenes, "evaluation_methods": [module.ORACLE],
            "required_methods": [module.ORACLE], "experiment_series": "formal_v2_t6_hopper",
            "include_cadr_oracle_age": True, "ack_lag": 1, "max_delay": 32,
        }
        episode_seeds = module.expected_episode_seeds(config)
        inputs = [{"role": "policy", "path": "private/checkpoint.pt", "sha256": "a" * 64}]
        references[seed] = {
            "config": config, "inputs": inputs, "runtime": runtime, "gpu": gpu,
            "package_versions": packages, "summary_sha256": "b" * 64,
            "receipt_sha256": "c" * 64, "config_sha256": "d" * 64,
            "results": {
                scene: result(module, scene, module.ORACLE, episode_seeds, 0.0, 1, 0)
                for scene in module.SCENARIOS
            },
        }
        for offset in (*module.OFFSETS, 0) if seed == 101 else module.OFFSETS:
            scenes = [module.PREFLIGHT_SCENARIO] if offset == 0 else list(module.SCENARIOS)
            run_id = f"seed{seed}_offset{offset}"
            config_path = tmp_path / f"{run_id}.json"
            config_hash = put(module, config_path, {
                **config,
                "scenarios": [{"name": scene} for scene in scenes],
                "evaluation_methods": [module.METHOD], "required_methods": [module.METHOD],
                "experiment_series": "ablation_v2_p2b1_hopper", "oracle_age_offset": offset,
                "include_cadr_oracle_age": False,
            })
            descriptor = {
                "run_id": run_id, "seed": seed, "offset": offset, "config": config_path.name,
                "config_sha256": config_hash, "scenarios": scenes, "inputs": inputs,
            }
            if offset == 0:
                manifest["preflight"] = descriptor
            else:
                manifest["runs"].append(descriptor)
            run_path = outputs / run_id
            run_path.mkdir(parents=True)
            summary = {
                "deployment_parameter_updates": False, "evaluation_methods": [module.METHOD],
                "oracle_age_offset": offset, "included_privileged_cadr_oracle_age_offset": True,
                "information_boundary": "privileged_selected_age_evaluation_only",
                "runtime": runtime, "local_source_provenance": {"tree_sha256": "e" * 64},
                "evaluations": {
                    scene: {
                        module.METHOD: result(
                            module, scene, module.METHOD, episode_seeds, float(offset),
                            1 if offset == 0 or seed == 101 else 2, offset,
                        ),
                    }
                    for scene in scenes
                },
            }
            digest = put(module, run_path / "summary.json", summary)
            digests[run_id] = {"summary_sha256": digest, "episodes": len(scenes) * module.EPISODES}
    manifest_hash = put(module, manifest_path, manifest)
    monkeypatch.setattr(module, "load_references", lambda _: references)
    preflight = module.verify_preflight(manifest_path, outputs, tmp_path)
    receipt = {
        "schema_version": 1,
        "phase": "complete",
        "identity": {
            "manifest_sha256": manifest_hash, "p2a_plan_sha256": module.P2A_PLAN_SHA256,
            "gpu": {**gpu, "uuid": "GPU-private"}, "package_versions": packages,
            "source_tree_sha256": "e" * 64, "git_commit": "f" * 40,
        },
        "preflight": preflight, "run_summaries": digests,
    }
    put(module, outputs / "execution_receipt.json", receipt)
    return manifest, manifest_path, outputs, output


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("returns", -0.0, "returns_bitwise_equal"),
        ("episode_seeds", 7, "invalid episode identity"),
        ("episode_lengths", 2, "episode_lengths_equal"),
    ],
)
def test_preflight_rejects_bit_pattern_seed_or_length_changes(module, staged, tmp_path, field, value, message):
    manifest, manifest_path, outputs, _ = staged
    summary_path = outputs / manifest["preflight"]["run_id"] / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    outcome = summary["evaluations"][module.PREFLIGHT_SCENARIO][module.METHOD]
    outcome[field][0] = value
    outcome["age_offset_diagnostics"]["decision_steps"] = sum(outcome["episode_lengths"])
    module.write_json(summary_path, summary)

    with pytest.raises(ValueError, match=message):
        module.verify_preflight(manifest_path, outputs, tmp_path)


def test_complete_verification_allows_changed_lengths_and_publishes_only_aggregates(module, staged, tmp_path):
    _, manifest_path, outputs, output = staged
    decision = module.aggregate(manifest_path, outputs, tmp_path, output)
    assert decision["p2b1_prediction_passed"] is True
    assert set(path.name for path in output.iterdir()) == {
        "absolute.csv", "paired.csv", "offset_diagnostics.csv", "manifest.json", "decision.json", "report.md",
    }
    evidence = (output / "manifest.json").read_text(encoding="utf-8")
    assert "GPU-private" not in evidence and "private/checkpoint" not in evidence
    assert '"episode_seeds":' not in evidence and '"returns":' not in evidence
    with (output / "offset_diagnostics.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    row = next(row for row in rows if row["offset"] == "2" and row["scenario"] == "fixed_delay_4")
    assert float(row["mean_applied_offset"]) == pytest.approx(80 / 60)
    assert float(row["fully_clipped_fraction"]) == pytest.approx(20 / 60)


def test_decision_requires_three_negative_decision_scenes_for_each_offset_pair(module):
    rows = [
        {"scenario": scene, "offset_left": -magnitude, "offset_right": magnitude, "ci95_high": -1}
        for scene in module.SCENARIOS for magnitude in (1, 2)
    ]
    for row in rows:
        if row["offset_right"] == 2 and row["scenario"] in ("fixed_delay_4", "fixed_delay_8"):
            row["ci95_high"] = 0
    decision = module.decision_for(rows)
    assert decision["offset_pairs"]["minus1_vs_plus1"]["passes"] is True
    assert decision["offset_pairs"]["minus2_vs_plus2"]["passes"] is False
    assert decision["p2b1_prediction_passed"] is False


def test_final_aggregation_rejects_unreceipted_summary_edits(module, staged, tmp_path):
    manifest, manifest_path, outputs, output = staged
    summary = outputs / manifest["runs"][0]["run_id"] / "summary.json"
    summary.write_text(summary.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.aggregate(manifest_path, outputs, tmp_path, output)
    assert not output.exists()
