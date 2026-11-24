from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_t6_batch_test", ROOT / "scripts/revision/run_t6_batch.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def gpu(uuid="GPU-test-A"):
    return {
        "index": "0",
        "uuid": uuid,
        "name": "Tesla T4",
        "memory.total": "15360",
        "driver_version": "580.82.07",
        "cuda_visible_devices": "",
    }


def test_resume_accepts_recycled_same_model_gpu_but_rejects_changed_inputs():
    saved = {
        "gpu": gpu(),
        "python": "/env/a/bin/python",
        "package_versions": {"torch": "2.8.0+cu128"},
        "manifest_sha256": "a" * 64,
        "source_tree_sha256": "b" * 64,
    }
    current = deepcopy(saved)
    current.update(gpu=gpu("GPU-test-B"), python="/env/b/bin/python")
    assert RUNNER.compatible_batch_identity(saved, current)

    for field, value in (
        ("name", "NVIDIA L4"),
        ("memory.total", "23034"),
        ("driver_version", "580.83.00"),
    ):
        different_gpu = current["gpu"] | {field: value}
        assert not RUNNER.compatible_batch_identity(saved, current | {"gpu": different_gpu})
    for field, value in (
        ("package_versions", {"torch": "2.9.0"}),
        ("manifest_sha256", "c" * 64),
        ("source_tree_sha256", "d" * 64),
    ):
        assert not RUNNER.compatible_batch_identity(saved, current | {field: value})


def test_clean_check_reports_exceptions_and_compares_float_bits():
    runs = [
        {"item": {"method_id": name, "clean_expected_equal": expected}}
        for name, expected in [
            ("direct", True), ("full_hazard", True), ("no_content", False),
        ]
    ]
    reference = {
        "episode_seeds": [7], "episode_lengths": [20], "returns": [0.0],
        "successes": [1.0], "successes_at_end": [1.0], "success_occupancies": [0.8],
    }
    results = {
        "direct": reference,
        "full_hazard": deepcopy(reference),
        "no_content": reference | {"returns": [-0.0], "success_occupancies": [0.7]},
    }
    clean = RUNNER.classify_clean(runs, results, "pickcube")
    assert clean["full_hazard"]["expected_equal"] is True
    assert clean["full_hazard"]["bitwise_equal"] is True
    assert clean["no_content"]["expected_equal"] is False
    assert clean["no_content"]["bitwise_equal"] is False
    assert clean["no_content"]["field_equal"]["returns"] is False
    assert clean["no_content"]["field_equal"]["success_occupancies"] is False


def test_native_pickcube_crash_retries_from_verified_snapshot(tmp_path, monkeypatch):
    config = {"eval_seed": 7}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    item = {
        "method_id": "direct", "evaluation_key": "direct",
        "config_sha256": RUNNER.sha256(config_path),
    }
    plan = {
        "item": item, "config": config, "config_path": config_path,
        "directory_name": "pickcube__seed101",
    }
    batch = {"task": "pickcube", "scenarios": ["clean_delay_0"], "eval_episodes": 1}
    args = SimpleNamespace(
        project_root=tmp_path, task="pickcube", python=Path("python"),
        maniskill_root=tmp_path / "ManiSkill",
    )
    inputs = {
        role: [(tmp_path / role, "0" * 64)] for role in ("policy", "dynamics", "dataset")
    }
    receipt_path = tmp_path / "batch_receipt.json"
    tree = "1" * 64
    current_gpu = gpu()
    receipt = {
        "identity": {"source_tree_sha256": tree},
        "active_gpu": current_gpu,
        "runs": {},
    }
    expected = {
        "scenario": "clean_delay_0", "episodes": 1, "privileged_selected_age": False,
        "episode_seeds": [7], "episode_lengths": [20], "returns": [1.0],
        "successes": [1.0], "successes_at_end": [1.0], "success_occupancies": [0.8],
    }
    launches = []

    def fake_subprocess(command, stdout, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        launches.append("--resume" in command)
        if len(launches) == 1:
            output.mkdir()
            (output / "config.json").write_text(json.dumps(config), encoding="utf-8")
            provenance = {
                name: {"path": str(inputs[role][0][0].resolve()), "sha256": "0" * 64}
                for name, role in (
                    ("policy_checkpoint", "policy"), ("dynamics_checkpoint", "dynamics"),
                    ("dataset", "dataset"),
                )
            }
            provenance["local_source_provenance"] = {"tree_sha256": tree}
            (output / "inputs.json").write_text(json.dumps(provenance), encoding="utf-8")
            stdout.write(b"Fatal Python error: gilstate_tss_set\n")
            return SimpleNamespace(returncode=-6)

        first = RUNNER.read_json(receipt_path)["runs"]["direct"]["attempts"][0]
        assert first["phase"] == "native_crash" and first["returncode"] == -6
        summary = {
            "local_source_provenance": {"tree_sha256": tree},
            "evaluation_methods": ["direct"],
            "evaluations": {"clean_delay_0": {"direct": expected}},
        }
        (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(RUNNER, "gpu_fingerprint", lambda: current_gpu)
    monkeypatch.setattr(RUNNER, "source_fingerprint", lambda _: tree)
    fake_process = SimpleNamespace(run=fake_subprocess, STDOUT=RUNNER.subprocess.STDOUT)
    monkeypatch.setattr(RUNNER, "subprocess", fake_process)
    actual = RUNNER.run_method(
        args, plan, inputs, batch, tmp_path, receipt_path, receipt, {}, -1
    )
    assert actual == expected
    assert launches == [False, True]
    completed = RUNNER.read_json(receipt_path)["runs"]["direct"]
    assert completed["phase"] == "complete"
    assert completed["attempt_count"] == 2
    assert completed["resume_count"] == completed["native_crash_count"] == 1
    assert completed["summary_sha256"] == RUNNER.sha256(tmp_path / completed["summary"])
