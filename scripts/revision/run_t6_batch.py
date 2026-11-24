"""Run one approved T6 task/seed batch in isolated evaluator processes."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import struct
import subprocess
import sys
from tempfile import NamedTemporaryFile
from typing import Any


MAX_ATTEMPTS = 3
CLEAN_SCENARIO = "clean_delay_0"
SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,159}")
DIGEST = re.compile(r"[0-9a-f]{64}")


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_receipt(path: Path, receipt: dict) -> None:
    receipt["updated_at"] = timestamp()
    with NamedTemporaryFile(
        dir=path.parent, prefix=".receipt-", suffix=".tmp",
        mode="w", encoding="utf-8", delete=False,
    ) as stream:
        scratch = Path(stream.name)
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    scratch.replace(path)


def safe_name(value: str, label: str) -> str:
    if not isinstance(value, str) or SAFE_NAME.fullmatch(value) is None:
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def relative_file(root: Path, value: str) -> Path:
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError(f"unsafe relative file path: {value!r}")
    file = root.joinpath(*path.parts)
    if not file.is_file():
        raise FileNotFoundError(file)
    return file


def output_path(root: Path, name: str) -> Path:
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError(f"unsafe output path: {path}")
    return path


def require_hash(path: Path, expected: str) -> None:
    if not isinstance(expected, str) or DIGEST.fullmatch(expected) is None:
        raise ValueError(f"invalid SHA-256 for {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, found {actual}")


def gpu_fingerprint() -> dict[str, str]:
    columns = ("index", "uuid", "name", "memory.total", "driver_version")
    request = subprocess.run(
        ["nvidia-smi", "--query-gpu=" + ",".join(columns), "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    rows = list(csv.reader(request.stdout.splitlines()))
    if len(rows) != 1 or len(rows[0]) != len(columns):
        raise ValueError("T6 batches require exactly one GPU visible to nvidia-smi")
    fingerprint = dict(zip(columns, (value.strip() for value in rows[0])))
    fingerprint["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return fingerprint


def compatible_batch_identity(saved: dict, current: dict) -> bool:
    # A recycled Colab VM can provide another physical card of the same type.
    # Paths and source/data/config identity still have to satisfy the evaluator's
    # saved snapshots. A different Python path is fine with identical versions.
    ignored = {"gpu", "python"}
    saved_non_gpu = {key: value for key, value in saved.items() if key not in ignored}
    current_non_gpu = {key: value for key, value in current.items() if key not in ignored}
    if saved_non_gpu != current_non_gpu:
        return False
    gpu_model_fields = ("name", "memory.total", "driver_version")
    return all(saved["gpu"][key] == current["gpu"][key] for key in gpu_model_fields)


def source_fingerprint(root: Path) -> str:
    package = root / "src" / "dt_corl_delay"
    files = sorted(path for path in package.rglob("*.py") if "__pycache__" not in path.parts)
    if not files:
        raise FileNotFoundError(f"CADR source tree missing: {package}")
    digest = hashlib.sha256()
    for source in files:
        name = source.relative_to(package).as_posix()
        digest.update(f"{name}\0{sha256(source)}\n".encode())
    return digest.hexdigest()


@contextmanager
def lock_batch(root: Path):
    # Linux/Colab uses a kernel lock that also survives an interrupted parent
    # while its evaluator still holds the file descriptor.
    import fcntl
    with output_path(root, ".batch.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another process owns the batch output: {root}") from error
        yield lock.fileno()


def load_plan(args: argparse.Namespace) -> tuple[dict, dict, list[dict]]:
    manifest = read_json(args.manifest)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported T6 manifest schema")
    selected = [
        batch for batch in manifest["batches"]
        if batch["task"] == args.task and batch["training_seed"] == args.seed
    ]
    if len(selected) != 1:
        raise ValueError("manifest must contain exactly one batch for the requested task and seed")
    batch = selected[0]
    safe_name(batch["batch_id"], "batch_id")
    episodes = batch["eval_episodes"]
    scenarios = batch["scenarios"]
    if type(episodes) is not int or episodes < 1:
        raise ValueError("eval_episodes must be a positive integer")
    if CLEAN_SCENARIO not in scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("batch scenarios must be unique and include clean_delay_0")
    input_files: dict[str, list[tuple[Path, str]]] = {}
    input_paths = set()
    for item in batch["inputs"]:
        file = relative_file(args.project_root, item["path"])
        if file in input_paths:
            raise ValueError(f"duplicate input: {item['path']}")
        input_paths.add(file)
        require_hash(file, item["sha256"])
        input_files.setdefault(item["role"], []).append((file, item["sha256"]))
    for role in ("policy", "dynamics", "dataset"):
        if len(input_files.get(role, [])) != 1:
            raise ValueError(f"batch requires one manifest input with role={role}")

    runs = []
    methods = set()
    output_names = set()
    paired = None
    for item in batch["runs"]:
        name = safe_name(item["method_id"], "method_id")
        key = safe_name(item["evaluation_key"], "evaluation_key")
        output_name = safe_name(item["output_basename"], "output_basename")
        if name in methods or output_name in output_names:
            raise ValueError(f"duplicate method or output name: {name}")
        if (name == "direct") != (key == "direct"):
            raise ValueError(f"direct method does not match its evaluator key: {name}")
        methods.add(name)
        output_names.add(output_name)
        config_path = relative_file(args.manifest.parent, item["config"])
        require_hash(config_path, item["config_sha256"])
        config = read_json(config_path)
        if config["seed"] != args.seed or config["eval_episodes"] != episodes:
            raise ValueError(f"wrong training seed or episode count in {config_path}")
        if config.get("evaluation_methods") != [key]:
            raise ValueError(f"invalid singleton evaluation_methods in {config_path}")
        if [scenario["name"] for scenario in config["scenarios"]] != scenarios:
            raise ValueError(f"scenario list differs from the batch manifest: {config_path}")
        device = config.get("device", "cuda")
        if device not in {"cuda", "cuda:0"}:
            raise ValueError(f"expected one Colab CUDA device: {config_path}")
        if args.task == "hopper":
            configured_dataset = relative_file(
                args.project_root,
                f"datasets/minari/{config['dataset_id']}/data/main_data.hdf5",
            )
        else:
            configured_dataset = args.project_root / config["dataset_path"]
        if configured_dataset.resolve() != input_files["dataset"][0][0].resolve():
            raise ValueError(f"config does not use the validated dataset: {config_path}")
        paired_signature = (
            config["eval_seed"], config["ack_lag"], config.get("environment"),
            config["dataset_id"],
            [
                {key: value for key, value in scenario.items() if key != "reference_delay_distribution"}
                for scenario in config["scenarios"]
            ],
        )
        if paired is None:
            paired = paired_signature
        elif paired_signature != paired:
            raise ValueError(f"pairing inputs differ across methods: {config_path}")
        expected = item["clean_expected_equal"]
        if type(expected) is not bool:
            raise ValueError(f"clean_expected_equal must be Boolean for {name}")
        runs.append({
            "item": item,
            "config": config,
            "config_path": config_path,
            "directory_name": f"{output_name}__{batch['batch_id']}",
        })
    if "direct" not in methods:
        raise ValueError("batch requires the shared direct reference")
    return batch, input_files, runs


def evaluator_command(args, run, inputs, output: Path) -> list[str]:
    paths = {role: str(inputs[role][0][0]) for role in ("policy", "dynamics")}
    if args.task == "hopper":
        module = "dt_corl_delay.evaluate_checkpoints"
        policy = ["--base-checkpoint", paths["policy"]]
    else:
        module = "dt_corl_delay.evaluate_maniskill_cadr_diffusion"
        policy = [
            "--policy-checkpoint", paths["policy"],
            "--source-root", str(args.maniskill_root),
        ]
    return [
        str(args.python), "-B", "-m", module,
        "--config", str(run["config_path"]),
        *policy, "--dynamics-checkpoint", paths["dynamics"], "--output", str(output),
    ]


def evaluator_environment(root: Path, project: Path) -> dict[str, str]:
    cache = output_path(root, "cache")
    cache.mkdir(exist_ok=True)
    fields = {
        "TMPDIR": "tmp", "TEMP": "tmp", "TMP": "tmp",
        "HF_HOME": "huggingface", "XDG_CACHE_HOME": "xdg",
        "XDG_CONFIG_HOME": "xdg-config", "TORCH_HOME": "torch",
        "CUDA_CACHE_PATH": "cuda", "MPLCONFIGDIR": "matplotlib",
        "MS_ASSET_DIR": "maniskill-assets",
    }
    variables = {}
    for name, child in fields.items():
        directory = output_path(cache, child)
        directory.mkdir(exist_ok=True)
        variables[name] = str(directory)
    return dict(
        os.environ, **variables,
        PYTHONPATH=str(project / "src"),
        PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
        DT_CORL_PROJECT_ROOT=str(project),
        MINARI_DATASETS_PATH=str(project / "datasets" / "minari"),
        DT_CORL_DATA_ROOT=str(project / "data" / "processed"),
        MUJOCO_GL="egl", MS_SKIP_ASSET_DOWNLOAD_PROMPT="1",
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
    )


def disable_core_dumps() -> None:
    # A native process crash must not create a core file in the source tree.
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def runtime_versions(python: Path) -> dict:
    code = """
from importlib.metadata import PackageNotFoundError, version
import json
import platform
packages = ("torch", "mujoco", "gymnasium", "minari", "mani-skill", "sapien", "h5py", "numpy")
versions = {"python": platform.python_version()}
for package in packages:
    try:
        versions[package] = version(package)
    except PackageNotFoundError:
        pass
print(json.dumps(versions))
"""
    output = subprocess.run(
        [str(python), "-I", "-B", "-c", code],
        check=True, capture_output=True, text=True,
    )
    return json.loads(output.stdout)


def verify_snapshot(root: Path, run: dict, inputs: dict, tree: str, task: str) -> None:
    if read_json(output_path(root, "config.json")) != run["config"]:
        raise ValueError(f"unexpected saved evaluation configuration: {root}")
    snapshot = read_json(output_path(root, "inputs.json"))
    roles = {"dynamics_checkpoint": "dynamics"}
    if task == "hopper":
        roles["base_checkpoint"] = "policy"
    else:
        roles.update(policy_checkpoint="policy", dataset="dataset")
    for key, role in roles.items():
        path, checksum = inputs[role][0]
        evidence = snapshot[key]
        if evidence["path"] != str(path.resolve()) or evidence["sha256"] != checksum:
            raise ValueError(f"{key} differs from the validated batch input: {root}")
    if snapshot["local_source_provenance"]["tree_sha256"] != tree:
        raise ValueError(f"saved evaluation used different source code: {root}")


def verify_summary(root: Path, run: dict, batch: dict, tree: str) -> dict:
    summary = read_json(output_path(root, "summary.json"))
    key = run["item"]["evaluation_key"]
    if summary.get("evaluation_methods") != [key]:
        raise ValueError(f"final summary does not declare exactly method {key}: {root}")
    if summary["local_source_provenance"]["tree_sha256"] != tree:
        raise ValueError(f"final summary has different source provenance: {root}")
    evaluations = summary["evaluations"]
    if set(evaluations) != set(batch["scenarios"]):
        raise ValueError(f"incomplete final scenarios: {root}")
    number = batch["eval_episodes"]
    seeds = [run["config"]["eval_seed"] + 100_003 * episode for episode in range(number)]
    privileged = run["item"]["method_id"] == "cadr_oracle_age"
    for scenario, methods in evaluations.items():
        if set(methods) - {key, "wall_seconds"} or key not in methods:
            raise ValueError(f"unexpected methods for {scenario} in {root}")
        result = methods[key]
        if (
            result["scenario"] != scenario or result["episodes"] != number
            or result["privileged_selected_age"] is not privileged
            or result["episode_seeds"] != seeds
            or len(result["episode_lengths"]) != number
            or len(result["returns"]) != number
            or not all(math.isfinite(value) for value in result["returns"])
        ):
            raise ValueError(f"incomplete or unpaired episodes for {scenario} in {root}")
        if batch["task"] == "pickcube":
            for field in ("successes", "successes_at_end", "success_occupancies"):
                if len(result[field]) != number or not all(
                    math.isfinite(value) for value in result[field]
                ):
                    raise ValueError(f"incomplete {field} for {scenario} in {root}")
    return evaluations[CLEAN_SCENARIO][key]


def classify_clean(runs: list[dict], results: dict, task: str) -> dict:
    reference = results["direct"]
    fields = ["episode_seeds", "episode_lengths", "returns"]
    if task == "pickcube":
        fields.extend(("successes", "successes_at_end", "success_occupancies"))
    comparison = {}
    for run in runs:
        item = run["item"]
        name = item["method_id"]
        result = results[name]
        by_field = {}
        for field in fields:
            left, right = reference[field], result[field]
            if field in {"episode_seeds", "episode_lengths"}:
                by_field[field] = left == right
            else:
                by_field[field] = all(
                    struct.pack(">d", a) == struct.pack(">d", b)
                    for a, b in zip(left, right)
                )
        comparison[name] = {
            "expected_equal": item["clean_expected_equal"],
            "bitwise_equal": all(by_field.values()),
            "field_equal": by_field,
        }
    return comparison


def log_tail(path: Path) -> str:
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 65_536))
        return stream.read().decode("utf-8", errors="replace")


def run_method(args, run, inputs, batch, root, receipt_path, receipt, env, lock_fd):
    item = run["item"]
    name = item["method_id"]
    directory = output_path(root, run["directory_name"])
    tree = receipt["identity"]["source_tree_sha256"]
    active_gpu = receipt["active_gpu"]
    records = receipt["runs"]
    if name not in records and directory.exists():
        raise FileExistsError(f"unrelated evaluation output exists: {directory}")
    record = records.setdefault(name, {"directory": directory.name, "attempts": []})
    if record["directory"] != directory.name:
        raise ValueError(f"evaluation directory changed for {name}")
    attempts = record["attempts"]
    if attempts and attempts[-1]["phase"] == "running":
        attempts[-1].update(phase="interrupted", ended_at=timestamp())
        write_receipt(receipt_path, receipt)

    if attempts and attempts[-1]["phase"] == "failed":
        raise RuntimeError(f"previous evaluator failed for {name}; inspect its preserved log")
    while not output_path(directory, "summary.json").is_file():
        if len(attempts) >= MAX_ATTEMPTS:
            raise RuntimeError(f"all {MAX_ATTEMPTS} evaluator attempts exhausted for {name}")
        if gpu_fingerprint() != active_gpu:
            raise RuntimeError("GPU fingerprint changed before evaluator launch")
        if source_fingerprint(args.project_root) != tree:
            raise RuntimeError("CADR source changed before evaluator launch")
        require_hash(run["config_path"], item["config_sha256"])
        resume = directory.exists()
        if resume:
            verify_snapshot(directory, run, inputs, tree, args.task)
        attempt_number = len(attempts) + 1
        log = output_path(root, f"{directory.name}.attempt-{attempt_number}.log")
        attempt = {
            "number": attempt_number, "phase": "running", "resume": resume,
            "log": log.name, "started_at": timestamp(), "gpu_uuid": active_gpu["uuid"],
        }
        attempts.append(attempt)
        record.update(
            phase="running", attempt_count=attempt_number,
            resume_count=sum(entry["resume"] for entry in attempts),
        )
        write_receipt(receipt_path, receipt)
        command = evaluator_command(args, run, inputs, directory)
        if resume:
            command.append("--resume")
        with log.open("xb") as stream:
            outcome = subprocess.run(
                command, cwd=args.project_root, env=env, stdout=stream,
                stderr=subprocess.STDOUT, pass_fds=(lock_fd,), check=False,
                preexec_fn=disable_core_dumps,
            )
        summary_exists = output_path(directory, "summary.json").is_file()
        tail = log_tail(log)
        crash = args.task == "pickcube" and "gilstate_tss_set" in tail.lower()
        attempt.update(returncode=outcome.returncode, native_crash=crash, ended_at=timestamp())
        record["native_crash_count"] = sum(
            entry.get("native_crash", False) for entry in attempts
        )
        if gpu_fingerprint() != active_gpu:
            attempt["phase"] = "failed"
            record["phase"] = "failed"
            write_receipt(receipt_path, receipt)
            raise RuntimeError("GPU fingerprint changed during evaluation")
        if outcome.returncode == 0 and summary_exists:
            attempt["phase"] = "completed"
        elif outcome.returncode != 0 and crash:
            attempt["phase"] = "native_crash"
        else:
            attempt["phase"] = "failed"
        record["phase"] = attempt["phase"]
        write_receipt(receipt_path, receipt)
        if attempt["phase"] == "failed":
            raise RuntimeError(
                f"{name} evaluator did not complete (exit {outcome.returncode}). See {log}"
            )

    verify_snapshot(directory, run, inputs, tree, args.task)
    result = verify_summary(directory, run, batch, tree)
    summary = output_path(directory, "summary.json")
    digest = sha256(summary)
    if record.get("summary_sha256", digest) != digest:
        raise ValueError(f"completed summary changed: {summary}")
    record.update(
        phase="complete", summary=summary.relative_to(root).as_posix(),
        summary_sha256=digest, attempt_count=len(attempts),
        resume_count=sum(attempt["resume"] for attempt in attempts),
        native_crash_count=sum(attempt.get("native_crash", False) for attempt in attempts),
    )
    write_receipt(receipt_path, receipt)
    return result


def run_batch(args: argparse.Namespace) -> Path:
    manifest_hash = sha256(args.manifest)
    batch, inputs, runs = load_plan(args)
    require_hash(args.manifest, manifest_hash)
    identity = {
        "manifest_sha256": manifest_hash, "task": args.task, "training_seed": args.seed,
        "batch_id": batch["batch_id"], "gpu": gpu_fingerprint(), "python": str(args.python),
        "package_versions": runtime_versions(args.python), "project_root": str(args.project_root),
        "maniskill_root": None if args.maniskill_root is None else str(args.maniskill_root),
        "source_tree_sha256": source_fingerprint(args.project_root),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    root = output_path(args.output, batch["batch_id"])
    root.mkdir(exist_ok=True)
    receipt_path = output_path(root, "batch_receipt.json")
    with lock_batch(root) as lock_fd:
        if receipt_path.is_file():
            receipt = read_json(receipt_path)
            if not compatible_batch_identity(receipt["identity"], identity):
                raise ValueError(f"batch provenance differs from saved receipt: {receipt_path}")
        else:
            receipt = {
                "schema_version": 1, "identity": identity, "runs": {},
                "scenarios": batch["scenarios"], "eval_episodes": batch["eval_episodes"],
                "inputs": batch["inputs"], "run_plan": batch["runs"],
                "started_at": timestamp(),
            }
        active_gpu = identity["gpu"]
        gpu_instances = receipt.setdefault("gpu_instances", [])
        if active_gpu not in gpu_instances:
            gpu_instances.append(active_gpu)
        receipt["active_gpu"] = active_gpu
        receipt["phase"] = "running"
        receipt.pop("error", None)
        write_receipt(receipt_path, receipt)
        try:
            env = evaluator_environment(root, args.project_root)
            clean = {}
            for run in runs:
                name = run["item"]["method_id"]
                clean[name] = run_method(
                    args, run, inputs, batch, root, receipt_path, receipt, env, lock_fd
                )
            for files in inputs.values():
                for file, digest in files:
                    require_hash(file, digest)
            for run in runs:
                require_hash(run["config_path"], run["item"]["config_sha256"])
            require_hash(args.manifest, manifest_hash)
            if source_fingerprint(args.project_root) != identity["source_tree_sha256"]:
                raise RuntimeError("CADR source changed while the batch was active")
            if gpu_fingerprint() != active_gpu:
                raise RuntimeError("GPU fingerprint changed while the batch was active")
            receipt["clean_checks"] = classify_clean(runs, clean, args.task)
            failed = [
                name for name, result in receipt["clean_checks"].items()
                if result["expected_equal"] and not result["bitwise_equal"]
            ]
            if failed:
                raise RuntimeError("clean-delay consistency failed: " + ", ".join(failed))
            receipt["phase"] = "complete"
            receipt["finished_at"] = timestamp()
        except BaseException as error:
            receipt["phase"] = "failed"
            receipt["error"] = f"{type(error).__name__}: {error}"
            write_receipt(receipt_path, receipt)
            raise
        write_receipt(receipt_path, receipt)
    return receipt_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("hopper", "pickcube"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--maniskill-root", type=Path)
    args = parser.parse_args()
    args.manifest = args.manifest.resolve(strict=True)
    args.project_root = args.project_root.resolve(strict=True)
    args.output = args.output.resolve()
    # Preserve a selected virtual environment's executable path through symlinks.
    args.python = args.python.absolute()
    if args.task == "pickcube":
        if args.maniskill_root is None:
            parser.error("--maniskill-root is required for PickCube")
        args.maniskill_root = args.maniskill_root.resolve(strict=True)
    try:
        receipt = run_batch(args)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError) as error:
        print(f"T6 batch failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    print(receipt)


if __name__ == "__main__":
    main()
