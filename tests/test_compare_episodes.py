import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "revision" / "compare_episodes.py"
FIELDS = (
    "series", "run_id", "task", "training_seed", "scenario", "method_key",
    "episode_index", "episode_seed", "return", "success",
)
HOPPER = "mujoco/hopper/medium-v0"
PICKCUBE = "PickCube-v1"


def row(index=0, *, seed=None, task=HOPPER, method="cadr_iql", value=100.0, success=""):
    return {
        "series": "formal_v1_minari_synthetic",
        "run_id": "archive",
        "task": task,
        "training_seed": 101,
        "scenario": "fixed_delay_4",
        "method_key": method,
        "episode_index": index,
        "episode_seed": 31000001 + 100003 * index if seed is None else seed,
        "return": value,
        "success": success,
    }


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_comparison(tmp_path, reference, candidate, *args, task=HOPPER):
    reference_path = tmp_path / "reference.csv"
    candidate_path = tmp_path / "candidate.csv"
    write_csv(reference_path, reference)
    write_csv(candidate_path, candidate)
    return subprocess.run(
        [
            sys.executable, str(SCRIPT), str(reference_path), str(candidate_path),
            "--task", task, "--training-seed", "101", *args,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_matches_reordered_rows_and_filters_methods(tmp_path):
    baseline = [row(0), row(1), row(method="direct_observation_cql"), row(task=PICKCUBE)]
    candidate = [dict(row(1), run_id="rerun"), dict(row(0), run_id="rerun")]
    output = tmp_path / "comparison.json"
    result = run_comparison(
        tmp_path, baseline, candidate, "--method", "cadr_iql", "--output", str(output)
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["passed"]
    assert summary["paired_count"] == 2
    assert summary["return_max_abs_delta"] == 0.0
    assert summary["success_pairs_with_values"] == 0
    assert len(json.loads(output.read_text(encoding="utf-8"))["episodes"]) == 2


def test_uses_paired_keys_and_flags_nonmatching_seeds(tmp_path):
    result = run_comparison(tmp_path, [row(0), row(1)], [row(0), row(1, seed=17)])
    assert result.returncode == 1
    summary = json.loads(result.stdout)["summary"]
    assert summary["paired_count"] == 1
    assert summary["missing_count"] == summary["extra_count"] == 1
    assert summary["extra_in_candidate"][0]["episode_seed"] == 17


def test_tolerates_only_return_difference(tmp_path):
    baseline = [row(task=PICKCUBE, value=1.0, success="1.0")]
    candidate = [row(task=PICKCUBE, value=1.000001, success=1)]
    strict = run_comparison(tmp_path, baseline, candidate, task=PICKCUBE)
    assert strict.returncode == 1
    result = run_comparison(tmp_path, baseline, candidate, "--abs-tol", "0.00001", task=PICKCUBE)
    assert result.returncode == 0
    summary = json.loads(result.stdout)["summary"]
    assert summary["return_nonzero_deltas"] == 1
    assert summary["return_max_abs_delta"] == pytest.approx(1e-6)
    assert summary["return_mean_delta"] == pytest.approx(1e-6)
    candidate[0]["success"] = 0
    failed = run_comparison(tmp_path, baseline, candidate, "--abs-tol", "0.00001", task=PICKCUBE)
    assert failed.returncode == 1
    assert json.loads(failed.stdout)["summary"]["success_mismatches"] == 1


@pytest.mark.parametrize("duplicate", [row(0), row(0, seed=17), row(1, seed=31000001)])
def test_rejects_duplicate_keys_or_group_indices_and_seeds(tmp_path, duplicate):
    result = run_comparison(tmp_path, [row(0)], [row(0), duplicate])
    assert result.returncode == 2
    assert "duplicate episode index or seed" in result.stderr


def test_empty_selection_cannot_pass(tmp_path):
    result = run_comparison(tmp_path, [row(task=PICKCUBE)], [row(task=PICKCUBE)])
    assert result.returncode == 1
    assert json.loads(result.stdout)["summary"]["paired_count"] == 0
