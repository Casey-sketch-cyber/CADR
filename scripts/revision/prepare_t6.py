"""Materialize the reviewed P2a configurations without changing v1 templates."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any


CORE = (
    "direct", "full_legacy", "full_hazard", "no_censoring", "in_order_age", "cadr_oracle_age",
)
HOPPER_EXTRA = (
    "no_content", "ack_map_hazard", "fixed_prior", "fixed_prior_matched", "single_model",
    "fixed_horizon_model_mean", "fixed_horizon_model_max_support", "full_hazard_state_mean",
)
TASK_SEEDS = {"hopper": (101, 102, 103, 104, 105), "pickcube": (101, 102)}
HOPPER_TEMPLATE = "generated_formal_v1_hopper_cadr_seed{seed}.json"
PICKCUBE_TEMPLATE = "generated_formal_v1_pickcube_synthetic_seed{seed}.json"
VARIANT_SETTINGS: dict[str, dict[str, Any]] = {
    "no_censoring": {"ablation_variant": "no_censoring", "use_censoring": False},
    "no_content": {"ablation_variant": "no_content", "use_content_conditioning": False,
                   "age_posterior": "legacy_cdf"},
    "ack_map_hazard": {"ablation_variant": "ack_map", "age_decision_rule": "map",
                       "aggregation": "map"},
    "fixed_prior": {"ablation_variant": "fixed_prior", "posterior_updates": False},
    "fixed_prior_matched": {"ablation_variant": "fixed_prior_matched", "posterior_updates": False},
    "single_model": {"ablation_variant": "single_model", "ensemble_members": [0]},
    "in_order_age": {"age_decision_rule": "in_order_age"},
    "fixed_horizon_model_mean": {"age_decision_rule": "fixed_horizon_model",
                                 "fixed_horizon_rule": "mean"},
    "fixed_horizon_model_max_support": {"age_decision_rule": "fixed_horizon_model",
                                        "fixed_horizon_rule": "max_support"},
    "full_hazard_state_mean": {"aggregation": "state_mean"},
    "cadr_oracle_age": {"include_cadr_oracle_age": True},
}
CLEAN_EXCEPTIONS = {
    "no_content": "Does not use observation content to detect fresh observations.",
    "fixed_horizon_model_mean": "Always uses the declared one-step nominal horizon.",
    "fixed_horizon_model_max_support": "Always uses the declared two-step nominal horizon.",
}


def method_settings(
    template: dict[str, Any], task: str, method: str,
) -> tuple[dict[str, Any], str]:
    is_dp = task == "pickcube"
    config = dict(
        template,
        include_oracle_upper_bound=False,
        include_cadr_oracle_age=False,
        diagnostic_float64_dynamics=False,
        measure_controller_timing=True,
    )
    if method not in {"direct", "full_legacy"}:
        config.update(age_posterior="hazard", age_decision_rule="posterior_mean",
                      use_content_conditioning=True, posterior_updates=True, use_censoring=True)
        config.update(VARIANT_SETTINGS.get(method, {}))
    series = "formal_v2" if method in {"direct", "full_legacy", "full_hazard", "cadr_oracle_age"} else "ablation_v2"
    config["experiment_series"] = f"{series}_t6_{task}"

    if method == "direct":
        key = "direct"
        label = "official_maniskill_dp_direct_delayed_observation" if is_dp else "direct_observation_iql"
    elif method == "cadr_oracle_age":
        key = label = "cadr_oracle_age"
    elif is_dp:
        variant = config.get("ablation_variant")
        if method == "in_order_age":
            variant = method
        key = f"cadr_dp_{variant}" if variant else "cadr_dp"
        label = (f"cadr_dp_ablation_{variant}_plus_official_maniskill_dp" if variant
                 else "cadr_posterior_state_reconstruction_plus_official_maniskill_dp")
    else:
        key = "cadr"
        label = (f"cadr_iql_ablation_{method}" if method in {
            "fixed_prior_matched", "in_order_age", "fixed_horizon_model_mean",
            "fixed_horizon_model_max_support",
        } else "cadr_iql")
    config["evaluation_methods"] = [key]
    config["required_methods"] = [label]
    return config, key


def read_index(path: Path, digest_column: str, size_column: str) -> dict[str, dict[str, Any]]:
    assets = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            name = row["server_relative_path"]
            digest = row[digest_column]
            if not name or not re.fullmatch(r"[0-9a-f]{64}", digest):
                continue
            if "role" in row and row["role"] != "preferred":
                continue
            asset = {"path": name, "sha256": digest}
            if row[size_column]:
                asset["bytes"] = int(row[size_column])
            if name in assets and assets[name] != asset:
                raise ValueError(f"conflicting recovery records for {name}")
            assets[name] = asset
    return assets


def read_data_index(index: Path) -> dict[str, dict[str, Any]]:
    original = index / "data_index.csv"
    data = read_index(original, "expected_sha256", "expected_bytes")
    supplemental = read_index(
        index / "supplemental_hashes.csv", "expected_sha256", "expected_bytes"
    )
    with original.open(encoding="utf-8-sig", newline="") as stream:
        author_rows = {row["server_relative_path"]: row for row in csv.DictReader(stream)}
    for name, record in supplemental.items():
        author = author_rows.get(name)
        if author is None or author["expected_sha256"]:
            raise ValueError(f"supplement may only fill a missing author-index hash: {name}")
        if not author["expected_bytes"] or record.get("bytes") != int(author["expected_bytes"]):
            raise ValueError(f"supplement differs from the author-index byte count: {name}")
    return data | supplemental


def batch_inputs(
    task: str, seed: int, template: dict[str, Any],
    weights: dict[str, dict[str, Any]], data: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if task == "hopper":
        dataset = f"datasets/minari/{template['dataset_id']}/data/main_data.hdf5"
        expected = {
            "policy": f"runs/formal_v1_iql_hopper_seed{seed}/checkpoint.pt",
            "dynamics": f"runs/formal_v1_cadr_hopper_seed{seed}/dynamics_checkpoint.pt",
            "dataset": dataset,
            "dataset_metadata": Path(dataset).with_name("metadata.json").as_posix(),
        }
    else:
        dataset = template["dataset_path"]
        expected = {
            "policy": f"runs/formal_v1_dp_pickcube_seed{seed}/checkpoints/best_eval_success_once.pt",
            "dynamics": f"runs/formal_v1_dp_dynamics_pickcube_seed{seed}/dynamics_checkpoint.pt",
            "dataset": dataset,
            "dataset_metadata": Path(dataset).with_suffix(".json").as_posix(),
        }
    return [{"role": role, **(weights if role in {"policy", "dynamics"} else data)[name]}
            for role, name in expected.items()]


def write_config(path: Path, value: dict[str, Any]) -> str:
    content = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"refusing to overwrite changed config: {path}")
    else:
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
    return sha256(content).hexdigest()


def prepare(project_root: Path, output: Path) -> dict[str, Any]:
    index = project_root / "docs/revision/handoff"
    weights = read_index(index / "checkpoint_index.csv", "recorded_sha256", "bytes")
    data = read_data_index(index)
    manifest: dict[str, Any] = {"schema_version": 1, "stage": "p2a", "batches": []}
    for task, seeds in TASK_SEEDS.items():
        template_pattern = HOPPER_TEMPLATE if task == "hopper" else PICKCUBE_TEMPLATE
        for seed in seeds:
            template_path = project_root / "configs" / template_pattern.format(seed=seed)
            content = template_path.read_bytes()
            template = json.loads(content)
            scenarios = [row["name"] for row in template["scenarios"]]
            expected_scenarios = 12 if task == "hopper" else 9
            if template["seed"] != seed or template["eval_episodes"] != 20 or (
                len(scenarios) != expected_scenarios or len(set(scenarios)) != expected_scenarios
                or "clean_delay_0" not in scenarios
            ):
                raise ValueError(f"invalid frozen P2a template: {template_path}")
            matrix = CORE + HOPPER_EXTRA if task == "hopper" and seed in (101, 102) else CORE
            runs = []
            for method in matrix:
                config, key = method_settings(template, task, method)
                relative = Path("configs/revision/T6") / task / f"seed{seed}" / f"{method}.json"
                digest = write_config(output / relative, config)
                runs.append(dict(method_id=method, evaluation_key=key, config=relative.as_posix(),
                                 config_sha256=digest, clean_expected_equal=method not in CLEAN_EXCEPTIONS,
                                 clean_exception=CLEAN_EXCEPTIONS.get(method),
                                 output_basename=f"revision_v2_t6_{task}_seed{seed}_{method}"))
            manifest["batches"].append(dict(
                task=task, training_seed=seed, batch_id=f"{task}_seed{seed}",
                source_config=template_path.relative_to(project_root).as_posix(),
                source_config_sha256=sha256(content).hexdigest(),
                scenarios=scenarios, eval_episodes=template["eval_episodes"],
                inputs=batch_inputs(task, seed, template, weights, data),
                expected_episodes=len(scenarios) * template["eval_episodes"] * len(runs), runs=runs,
            ))
    manifest["method_runs"] = sum(len(batch["runs"]) for batch in manifest["batches"])
    manifest["expected_episodes"] = sum(batch["expected_episodes"] for batch in manifest["batches"])
    write_config(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = prepare(args.project_root.resolve(), args.output.resolve())
    print(json.dumps({key: plan[key] for key in ("stage", "method_runs", "expected_episodes")}))


if __name__ == "__main__":
    main()
