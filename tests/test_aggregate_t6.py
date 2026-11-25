import importlib
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/revision"))
    return importlib.import_module("aggregate_t6")


def test_bootstrap_keeps_episode_pairs_and_checks_recovery_denominator(module):
    direct = np.arange(40, dtype=float).reshape(2, 20)
    clean = direct + 10
    method = direct + 15
    means, draws = module.means_and_bootstrap(np.stack((method, direct, clean), axis=-1))
    result = module.recovery(means, draws, "fixed_delay_4")
    assert result["recovery_pct"] == pytest.approx(150)
    assert result["recovery_ci95_low"] == pytest.approx(150)
    assert result["recovery_ci95_high"] == pytest.approx(150)
    assert module.recovery(means, draws, module.CLEAN)["recovery_pct"] is None

    clean = direct + np.array([[-1], [2]])
    means, draws = module.means_and_bootstrap(np.stack((method, direct, clean), axis=-1))
    result = module.recovery(means, draws, "fixed_delay_4")
    assert result["recovery_status"] == "bootstrap_denominator_unstable"
    assert 0 < result["recovery_positive_denominator_fraction"] < 1
    assert result["recovery_ci95_low"] is None


def test_cohorts_do_not_pool_gpu_strata_and_keep_required_seed_count(module):
    task, endpoint, scene = "hopper", "raw_return", "fixed_delay_4"
    values = np.arange(20, dtype=float)
    batches = {
        (task, seed): {
            "metrics": {("direct", scene): {endpoint: values}},
            "evidence": {"stratum": stratum, "gpu_model": gpu},
        }
        for seed, stratum, gpu in [(101, "a", "L4"), (102, "b", "T4"), (103, "a", "L4")]
    }
    groups = module.cohorts(batches, task, (101, 102, 103), (("direct", scene, endpoint),))
    assert [(info["gpu_model"], info["training_seeds"], rows.shape) for info, rows in groups] == [
        ("L4", "101 103", (2, 20, 1)), ("T4", "102", (1, 20, 1)),
    ]
    assert all(info["scope"] == "incomplete_gpu_stratum" for info, _ in groups)
    assert module.expected_seeds("hopper", "full_hazard", "in_order_age") == (101, 102, 103, 104, 105)
    assert module.expected_seeds("hopper", "full_hazard", "fixed_prior") == (101, 102)


def test_decision_uses_hopper_priority_intervals(module):
    rows = [
        {"contrast": contrast, "scenario": scene, "scope": "complete_same_gpu", "endpoint": "raw_return",
         "seed_count": 5, "interval_direction": "includes_zero", "delta": 1, "ci95_low": -1, "ci95_high": 2}
        for contrast in module.CORE_CONTRASTS for scene in module.SCENARIOS["hopper"]
    ]
    outcome = module.assess_decision(rows, [])
    assert outcome["route_b_discussion_flag"] and not outcome["route_a_criteria_met"]
    for row in rows:
        if ((row["contrast"] == "hazard_vs_in_order" and row["scenario"] in module.PRIORITY[:2])
                or (row["contrast"] == "censoring" and row["scenario"] == "ood_support_shift")):
            row.update(interval_direction="positive", ci95_low=0.5)
    outcome = module.assess_decision(rows, [])
    assert outcome["route_a_criteria_met"] and outcome["censoring_title_criterion_met"]
    assert not outcome["route_b_discussion_flag"]

    fixed_legacy = next(row for row in rows if row["contrast"] == "hazard_vs_legacy"
                        and row["scenario"] == "fixed_delay_1")
    fixed_legacy.update(interval_direction="negative", delta=-1, ci95_low=-2, ci95_high=-0.5)
    outcome = module.assess_decision(rows, [])
    assert outcome["priority_positive_hazard_vs_in_order"] == list(module.PRIORITY[:2])
    assert outcome["all_clearly_negative_hazard_vs_legacy"] == ["fixed_delay_1"]
    assert not outcome["route_a_criteria_met"]
    assert module.assess_decision(rows[:-1], [])["status"] == "insufficient_complete_homogeneous_hopper_data"


def test_missing_batches_report_incomplete_without_episode_artifacts(module, tmp_path):
    prepare = importlib.import_module("prepare_t6")
    plan = tmp_path / "plan"
    prepare.prepare(ROOT, plan)
    aggregate = tmp_path / "aggregate"
    result = module.aggregate(plan / "manifest.json", tmp_path / "outputs", aggregate)
    assert result["verified_episodes"] == 0 and result["expected_episodes"] == 13200
    assert {batch["status"] for batch in result["batches"]} == {"missing"}
    assert {path.name for path in aggregate.iterdir()} == {
        "absolute_hopper.csv", "absolute_pickcube.csv", "clean_checks.csv", "decision.json",
        "historical_d4rl_hopper.csv", "manifest.json", "paired_hopper.csv", "paired_pickcube.csv",
        "report.md", "timing.csv",
    }
    decision = json.loads((aggregate / "decision.json").read_text(encoding="utf-8"))
    assert decision["status"] == "insufficient_complete_homogeneous_hopper_data"
