#!/usr/bin/env python3
"""Run the P2b-1 schedule on a single Colab GPU after an exact zero-offset check."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from aggregate_p2b1 import load_references, verify_preflight
from prepare_p2b1 import (
    DECISION_SCENARIOS, DESCRIPTIVE_SCENARIOS, EVAL_EPISODES, METHOD, OFFSETS,
    P2A_PLAN_SHA256, PREFLIGHT_SCENARIOS, SEEDS,
)
from run_t6_batch import (
    disable_core_dumps, evaluator_environment, gpu_fingerprint, lock_batch,
    output_path, read_json, relative_file, require_hash, runtime_versions, safe_name,
    sha256, source_fingerprint, verify_snapshot, write_receipt,
)

OFFSET_LABELS = dict(OFFSETS) | {0: "0"}
SCENARIOS = frozenset(DECISION_SCENARIOS + DESCRIPTIVE_SCENARIOS)


def load_schedule(args: argparse.Namespace) -> tuple[dict, dict, list[dict]]:
    plan = read_json(args.manifest)
    if (plan.get("schema_version"), plan.get("stage"), plan.get("eval_episodes")) != (1, "p2b1", EVAL_EPISODES):
        raise ValueError("unsupported P2b-1 plan")
    if (plan.get("new_episodes"), plan.get("reference_episodes"), plan.get("preflight_episodes")) != (800, 200, 20):
        raise ValueError("unexpected P2b-1 episode budget")
    if plan.get("p2a_plan_sha256") != P2A_PLAN_SHA256:
        raise ValueError("P2b-1 must use the published P2a plan")
    if (plan.get("decision_scenarios") != list(DECISION_SCENARIOS)
        or plan.get("descriptive_scenarios") != list(DESCRIPTIVE_SCENARIOS)):
        raise ValueError("P2b-1 decision and descriptive scenes do not match the protocol")
    reference = args.reference / "inventory/manifest.json"
    require_hash(reference, P2A_PLAN_SHA256)
    p2a = read_json(reference)
    references = {b["training_seed"]: b for b in p2a["batches"]
                  if b["task"] == "hopper" and b["training_seed"] in SEEDS}
    expected = {(seed, offset) for seed in SEEDS for offset, _ in OFFSETS}
    if len(plan["runs"]) != len(expected) or {(r["seed"], r["offset"]) for r in plan["runs"]} != expected:
        raise ValueError("P2b-1 must contain all eight unique signed-offset runs")
    if (plan["preflight"]["seed"], plan["preflight"]["offset"]) != (101, 0):
        raise ValueError("P2b-1 preflight must be the seed-101 zero offset")

    validated = set()

    def bind(descriptor: dict) -> dict:
        seed, offset = descriptor["seed"], descriptor["offset"]
        if type(seed) is not int or type(offset) is not int:
            raise ValueError("P2b-1 seed and offset must be integers")
        run_id = safe_name(descriptor["run_id"], "run_id")
        if run_id != f"hopper_seed{seed}_offset_{OFFSET_LABELS[offset]}":
            raise ValueError(f"P2b-1 run disagrees with its identifier: {run_id}")
        archived = references[seed]
        if descriptor["inputs"] != archived["inputs"]:
            raise ValueError(f"frozen input inventory differs from P2a: {run_id}")
        inputs = {}
        for item in descriptor["inputs"]:
            path = relative_file(args.project_root, item["path"])
            identity = (path, item["sha256"])
            if identity not in validated:
                require_hash(*identity)
                validated.add(identity)
            inputs.setdefault(item["role"], []).append((path, item["sha256"]))
        reference_config = next(r for r in archived["runs"] if r["method_id"] == "cadr_oracle_age")
        baseline_path = relative_file(reference.parent, reference_config["config"])
        require_hash(baseline_path, reference_config["config_sha256"])
        config_path = relative_file(args.manifest.parent, descriptor["config"])
        require_hash(config_path, descriptor["config_sha256"])
        config = read_json(config_path)
        baseline = read_json(baseline_path)
        scenes = SCENARIOS if offset != 0 else set(PREFLIGHT_SCENARIOS)
        baseline.update(
            scenarios=[s for s in baseline["scenarios"] if s["name"] in scenes],
            experiment_series="ablation_v2_p2b1_hopper", include_cadr_oracle_age=False,
            oracle_age_offset=offset, evaluation_methods=[METHOD], required_methods=[METHOD],
        )
        # The complete config comparison covers the paired seeds and channel definitions.
        if config != baseline or descriptor["scenarios"] != [s["name"] for s in baseline["scenarios"]]:
            raise ValueError(f"P2b-1 config differs from the reviewed P2a oracle settings: {run_id}")
        return {"item": descriptor, "config_path": config_path, "config": config, "inputs": inputs}

    return plan, bind(plan["preflight"]), [bind(r) for r in plan["runs"]]


def run_method(args, run, identity, receipt, lock_fd, env):
    item = run["item"]
    run_id = item["run_id"]
    path = output_path(args.outputs, run_id)
    summary = output_path(path, "summary.json")
    tree = identity["source_tree_sha256"]
    if gpu_fingerprint() != identity["gpu"] or source_fingerprint(args.project_root) != tree:
        raise RuntimeError("GPU or source changed during the P2b-1 session")
    require_hash(run["config_path"], item["config_sha256"])
    if path.exists():
        verify_snapshot(path, run, run["inputs"], tree, "hopper")
    if not summary.is_file():
        command = [
            str(args.python), "-B", "-m", "dt_corl_delay.evaluate_checkpoints",
            "--config", str(run["config_path"]),
            "--base-checkpoint", str(run["inputs"]["policy"][0][0]),
            "--dynamics-checkpoint", str(run["inputs"]["dynamics"][0][0]),
            "--output", str(path),
        ]
        if path.exists():
            command.append("--resume")
        with output_path(args.outputs, run_id + ".log").open("ab") as log:
            subprocess.run(command, cwd=args.project_root, env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True, pass_fds=(lock_fd,),
                           preexec_fn=disable_core_dumps)
    verify_snapshot(path, run, run["inputs"], tree, "hopper")
    document = read_json(summary)
    if (set(document["evaluations"]) != set(item["scenarios"])
        or document.get("evaluation_methods") != [METHOD]
        or document.get("oracle_age_offset") != item["offset"]
        or document["local_source_provenance"]["tree_sha256"] != tree):
        raise ValueError(f"Invalid P2b-1 evaluation summary: {run_id}")
    entries = [row[METHOD] for row in document["evaluations"].values()]
    if any(len(row["returns"]) != EVAL_EPISODES or row["privileged_selected_age"] is not True for row in entries):
        raise ValueError(f"Wrong P2b-1 rollout count or information boundary: {run_id}")
    runtime = document["runtime"]
    if runtime["device"] != identity["gpu"]["name"] or runtime["torch"] != identity["package_versions"]["torch"]:
        raise ValueError(f"Wrong P2b-1 evaluator runtime: {run_id}")
    record = {"summary_sha256": sha256(summary), "episodes": len(entries) * EVAL_EPISODES}
    previous = receipt["run_summaries"].get(run_id)
    if previous is not None and previous != record:
        raise ValueError(f"Previously verified summary changed: {run_id}")
    receipt["run_summaries"][run_id] = record


def run_schedule(args: argparse.Namespace) -> Path:
    manifest_sha256 = sha256(args.manifest)
    _, preflight, runs = load_schedule(args)
    require_hash(args.manifest, manifest_sha256)
    gpu = gpu_fingerprint()
    package_versions = runtime_versions(args.python)
    baseline = load_references(args.reference)[SEEDS[0]]
    expected_gpu, expected_packages = baseline["gpu"], baseline["package_versions"]
    if any(gpu[key] != value for key, value in expected_gpu.items()) or package_versions != expected_packages:
        raise ValueError("the Colab GPU or packages differ from the archived P2a Hopper runtime")
    identity = {
        "manifest_sha256": manifest_sha256, "p2a_plan_sha256": P2A_PLAN_SHA256,
        "gpu": gpu, "package_versions": package_versions,
        "source_tree_sha256": source_fingerprint(args.project_root),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=args.project_root, text=True,
        ).strip(),
    }
    args.outputs.mkdir(parents=True, exist_ok=True)
    receipt_path = output_path(args.outputs, "execution_receipt.json")
    with lock_batch(args.outputs) as lock_fd:
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt["identity"] != identity:
                raise ValueError("continuing P2b-1 requires the same Colab session and inputs")
        else:
            receipt = {"schema_version": 1, "identity": identity, "run_summaries": {}}
        receipt.pop("error", None)
        receipt["phase"] = "preflight"
        write_receipt(receipt_path, receipt)
        try:
            env = evaluator_environment(args.outputs, args.project_root)
            run_method(args, preflight, identity, receipt, lock_fd, env)
            receipt["preflight"] = verify_preflight(args.manifest, args.outputs, args.reference)
            receipt["phase"] = "running"
            write_receipt(receipt_path, receipt)
            for run in runs:
                receipt["active_run"] = run["item"]["run_id"]
                write_receipt(receipt_path, receipt)
                run_method(args, run, identity, receipt, lock_fd, env)
                write_receipt(receipt_path, receipt)
            require_hash(args.manifest, identity["manifest_sha256"])
            if gpu_fingerprint() != gpu or source_fingerprint(args.project_root) != identity["source_tree_sha256"]:
                raise RuntimeError("GPU or source changed before P2b-1 completion")
            for path, digest in {pair for run in runs for records in run["inputs"].values() for pair in records}:
                require_hash(path, digest)
            receipt.pop("active_run", None)
            receipt["phase"] = "complete"
        except BaseException as error:
            receipt.update(phase="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            write_receipt(receipt_path, receipt)
    return receipt_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "project-root", "reference", "outputs", "python"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    for name in ("manifest", "project_root", "reference", "outputs"):
        setattr(args, name, getattr(args, name).resolve())
    args.python = args.python.absolute()
    print(run_schedule(args))


if __name__ == "__main__":
    main()
