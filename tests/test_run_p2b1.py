"""Check that the P2b schedule starts signed runs only after the zero gate."""

from argparse import Namespace
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def scheduler(monkeypatch, tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts" / "revision"
    monkeypatch.syspath_prepend(str(scripts))
    module = importlib.import_module("run_p2b1")
    plan = tmp_path / "manifest.json"
    plan.write_text("{}", encoding="utf-8")
    args = Namespace(manifest=plan, project_root=tmp_path,
                     reference=tmp_path / "reference", outputs=tmp_path / "outputs",
                     python=Path("python"))
    gpu = {"name": "Tesla T4", "memory.total": "15360", "driver_version": "580.82.07"}
    packages = {"torch": "2.8.0+cu128"}

    def item(name):
        return {"item": {"run_id": name}, "inputs": {}}

    monkeypatch.setattr(module, "load_schedule", lambda _: ({}, item("zero"), [item("signed")]))
    monkeypatch.setattr(module, "gpu_fingerprint", lambda: gpu)
    monkeypatch.setattr(module, "runtime_versions", lambda _: packages)
    monkeypatch.setattr(module, "load_references", lambda _: {101: {"gpu": gpu, "package_versions": packages}})
    monkeypatch.setattr(module, "source_fingerprint", lambda _: "source")
    monkeypatch.setattr(module.subprocess, "check_output", lambda *_, **__: "commit")
    seen = []

    def evaluate(_args, run, _identity, receipt, _lock, _env):
        name = run["item"]["run_id"]
        seen.append(name)
        receipt["run_summaries"][name] = {"summary_sha256": "hash", "episodes": 20}

    monkeypatch.setattr(module, "run_method", evaluate)
    return module, args, seen


def test_preflight_mismatch_blocks_signed_rollouts(scheduler, monkeypatch):
    module, args, seen = scheduler

    def mismatch(*_):
        raise ValueError("zero offset differs")

    monkeypatch.setattr(module, "verify_preflight", mismatch)
    with pytest.raises(ValueError, match="zero offset differs"):
        module.run_schedule(args)
    assert seen == ["zero"]
    receipt = module.read_json(args.outputs / "execution_receipt.json")
    assert receipt["phase"] == "failed"
    assert list(receipt["run_summaries"]) == ["zero"]


def test_verified_preflight_is_saved_before_signed_rollouts(scheduler, monkeypatch):
    module, args, seen = scheduler

    def verify(*_):
        assert seen == ["zero"]
        seen.append("verified")
        return {"bitwise_equal": True}

    monkeypatch.setattr(module, "verify_preflight", verify)
    receipt = module.read_json(module.run_schedule(args))
    assert seen == ["zero", "verified", "signed"]
    assert receipt["phase"] == "complete"
    assert receipt["preflight"] == {"bitwise_equal": True}
