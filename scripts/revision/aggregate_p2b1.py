"""Verify and summarize the preregistered Hopper age-offset diagnostic."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re

import numpy as np

from aggregate_t6 import (
    BOOTSTRAP_SEED,
    RESAMPLES,
    interval,
    load_json,
    means_and_bootstrap,
    relative_file,
    validate_plan as validate_p2a_plan,
    write_csv,
    write_json,
)
from prepare_p2b1 import (
    DECISION_SCENARIOS,
    DESCRIPTIVE_SCENARIOS,
    EVAL_EPISODES as EPISODES,
    METHOD,
    OFFSETS as PLANNED_OFFSETS,
    P2A_PLAN_SHA256,
    PREFLIGHT_SCENARIOS,
    SEEDS,
)


P2A_EVIDENCE_SHA256 = "0dffc0060bfd3e7a6c14708780bcd468fe4799d3f813a68f363286e89b270c38"
P2A_EVIDENCE = Path(__file__).resolve().parents[2] / "results/revision/T6/manifest.json"
OFFSETS = tuple(offset for offset, _ in PLANNED_OFFSETS)
SCENARIOS = DECISION_SCENARIOS + DESCRIPTIVE_SCENARIOS
PREFLIGHT_SCENARIO, = PREFLIGHT_SCENARIOS
ORACLE = "cadr_oracle_age"
NEW_EPISODES = len(SEEDS) * len(OFFSETS) * len(SCENARIOS) * EPISODES
REFERENCE_EPISODES = len(SEEDS) * len(SCENARIOS) * EPISODES
GPU_FIELDS = ("name", "memory.total", "driver_version")
RUNTIME_FIELDS = ("device", "cuda", "torch", "dynamics_rollout_dtype")


def load_plan(path: Path) -> tuple[dict, str]:
    plan, digest = load_json(path)
    expected = {
        "schema_version": 1, "stage": "p2b1", "p2a_plan_sha256": P2A_PLAN_SHA256,
        "eval_episodes": EPISODES, "new_episodes": NEW_EPISODES,
        "reference_episodes": REFERENCE_EPISODES, "preflight_episodes": EPISODES,
        "decision_scenarios": list(DECISION_SCENARIOS),
        "descriptive_scenarios": list(DESCRIPTIVE_SCENARIOS),
    }
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("P2b-1 manifest differs from the reviewed design")
    descriptors = [*plan["runs"], plan["preflight"]]
    ids = [item["run_id"] for item in descriptors]
    pairs = [(item["seed"], item["offset"]) for item in plan["runs"]]
    if (
        len(ids) != len(set(ids))
        or any(not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", run_id) for run_id in ids)
        or len(pairs) != len(SEEDS) * len(OFFSETS)
        or set(pairs) != {(seed, offset) for seed in SEEDS for offset in OFFSETS}
        or (plan["preflight"]["seed"], plan["preflight"]["offset"]) != (SEEDS[0], 0)
    ):
        raise ValueError("P2b-1 manifest has missing, duplicate or unexpected run identities")
    for item in descriptors:
        scenes = item["scenarios"]
        expected_scenes = (PREFLIGHT_SCENARIO,) if item is plan["preflight"] else SCENARIOS
        if (
            type(item["seed"]) is not int or type(item["offset"]) is not int
            or len(scenes) != len(expected_scenes) or set(scenes) != set(expected_scenes)
        ):
            raise ValueError("P2b-1 run has unexpected seed, offset or scenario coverage")
    return plan, digest


def load_public_evidence() -> dict:
    # Git working-tree newline conversion does not change the published blob.
    content = P2A_EVIDENCE.read_bytes().replace(b"\r\n", b"\n")
    if sha256(content).hexdigest() != P2A_EVIDENCE_SHA256:
        raise ValueError("published P2a evidence does not match the reviewed digest")
    return json.loads(content)


def result_arrays(result: dict, scenario: str, expected_seeds: list[int], method: str) -> np.ndarray:
    lengths = result["episode_lengths"]
    seeds = result["episode_seeds"]
    returns = result["returns"]
    if (
        result["scenario"] != scenario or result["method"] != method
        or result["privileged_selected_age"] is not True or result["episodes"] != EPISODES
        or len(seeds) != EPISODES or any(type(seed) is not int for seed in seeds)
        or seeds != expected_seeds
        or len(lengths) != EPISODES or any(type(length) is not int or length < 1 for length in lengths)
        or len(returns) != EPISODES or any(type(value) not in (int, float) for value in returns)
    ):
        raise ValueError(f"invalid episode identity or privileged result for {method}/{scenario}")
    vector = np.asarray(returns, dtype="<f8")
    if not np.isfinite(vector).all():
        raise ValueError(f"non-finite returns for {method}/{scenario}")
    return vector


def expected_episode_seeds(config: dict) -> list[int]:
    return [config["eval_seed"] + 100_003 * episode for episode in range(EPISODES)]


def load_references(reference: Path) -> dict[int, dict]:
    inventory = reference.resolve() / "inventory"
    archived_plan, _ = load_json(inventory / "manifest.json", P2A_PLAN_SHA256)
    p2a_batches = validate_p2a_plan(archived_plan)
    published = load_public_evidence()
    if published["plan_sha256"] != P2A_PLAN_SHA256 or published["verified_episodes"] != 13200:
        raise ValueError("published P2a evidence has unexpected source or coverage")
    evidence = {
        row["training_seed"]: row for row in published["batches"]
        if row["task"] == "hopper" and row["training_seed"] in SEEDS
    }
    references = {}
    for seed in SEEDS:
        batch = p2a_batches["hopper", seed]
        public = evidence[seed]
        public_oracle = public["methods"][ORACLE]
        config_item = next(item for item in batch["runs"] if item["method_id"] == ORACLE)
        config, _ = load_json(
            relative_file(inventory, config_item["config"]), config_item["config_sha256"],
        )
        root = reference.resolve() / "outputs" / batch["batch_id"]
        receipt, receipt_hash = load_json(root / "batch_receipt.json", public["receipt_sha256"])
        identity = receipt["identity"]
        record = receipt["runs"][ORACLE]
        summary, summary_hash = load_json(
            relative_file(root, record["summary"]), public_oracle["summary_sha256"],
        )
        runtime = {key: summary["runtime"][key] for key in RUNTIME_FIELDS}
        gpu = {key: identity["gpu"][key] for key in GPU_FIELDS}
        if (
            public["status"] != "complete" or receipt["phase"] != "complete"
            or identity["batch_id"] != batch["batch_id"] or identity["task"] != "hopper"
            or identity["training_seed"] != seed or identity["manifest_sha256"] != P2A_PLAN_SHA256
            or receipt["inputs"] != batch["inputs"] or receipt["run_plan"] != batch["runs"]
            or config_item["config_sha256"] != public_oracle["config_sha256"]
            or record["summary_sha256"] != summary_hash or record["phase"] != "complete"
            or config["seed"] != seed or config["eval_episodes"] != EPISODES
            or config["evaluation_methods"] != [ORACLE] or summary["evaluation_methods"] != [ORACLE]
            or summary["deployment_parameter_updates"] is not False
            or summary["local_source_provenance"]["tree_sha256"] != identity["source_tree_sha256"]
            or gpu["name"] != public["gpu_model"] or gpu["driver_version"] != public["driver"]
            or gpu["memory.total"] != public["gpu_memory_mib"]
            or identity["package_versions"] != public["package_versions"]
        ):
            raise ValueError(f"invalid P2a oracle reference provenance for seed {seed}")
        results = {name: summary["evaluations"][name][ORACLE] for name in SCENARIOS}
        for name, result in results.items():
            result_arrays(result, name, expected_episode_seeds(config), public_oracle["evaluator_method"])
        references[seed] = {
            "config": config, "inputs": batch["inputs"], "results": results,
            "summary_sha256": summary_hash, "receipt_sha256": receipt_hash,
            "config_sha256": config_item["config_sha256"], "runtime": runtime,
            "gpu": gpu, "package_versions": identity["package_versions"],
        }
    for field in ("runtime", "gpu", "package_versions"):
        if references[101][field] != references[102][field]:
            raise ValueError(f"P2a oracle reference has mixed {field}")
    return references


def validate_diagnostics(result: dict, offset: int) -> dict:
    diagnostics = result["age_offset_diagnostics"]
    steps, offset_sum, clipped = (
        diagnostics[key] for key in ("decision_steps", "effective_offset_sum", "fully_clipped_steps")
    )
    if (
        any(type(count) is not int for count in (steps, offset_sum, clipped))
        or steps != sum(result["episode_lengths"]) or not 0 <= clipped <= steps
    ):
        raise ValueError(f"inconsistent age-offset counts for {result['scenario']}")
    nonzero_steps = steps - clipped
    signed_sum = -offset_sum if offset < 0 else offset_sum
    valid_sum = (
        nonzero_steps <= signed_sum <= abs(offset) * nonzero_steps
        if offset else offset_sum == clipped == 0
    )
    if (
        not valid_sum
        or not math.isclose(diagnostics["mean_applied_offset"], offset_sum / steps, abs_tol=1e-12)
        or not math.isclose(diagnostics["fully_clipped_fraction"], clipped / steps, abs_tol=1e-12)
    ):
        raise ValueError(f"inconsistent age-offset rates for {result['scenario']}")
    return diagnostics


def load_run(item: dict, manifest_path: Path, outputs: Path, reference: dict,
             expected_hash: str | None = None) -> dict:
    name, seed, offset = (item[key] for key in ("run_id", "seed", "offset"))
    config, _ = load_json(relative_file(manifest_path.resolve().parent, item["config"]), item["config_sha256"])
    frozen, scenarios = reference["config"], item["scenarios"]
    base_scenes = {scene["name"]: scene for scene in frozen["scenarios"]}
    expected_config = dict(
        frozen, scenarios=[base_scenes[scene] for scene in scenarios],
        experiment_series="ablation_v2_p2b1_hopper", oracle_age_offset=offset,
        evaluation_methods=[METHOD], required_methods=[METHOD], include_cadr_oracle_age=False,
    )
    if (
        config != expected_config or type(config["oracle_age_offset"]) is not int
        or config["include_cadr_oracle_age"] is not False or item["inputs"] != reference["inputs"]
    ):
        raise ValueError(f"P2b-1 config changes frozen reference settings for {name}")
    summary, digest = load_json(relative_file(outputs.resolve(), f"{name}/summary.json"), expected_hash)
    runtime = {key: summary["runtime"][key] for key in RUNTIME_FIELDS}
    if (
        summary["deployment_parameter_updates"] is not False
        or summary["evaluation_methods"] != [METHOD]
        or summary.get("included_privileged_cadr_oracle_age_offset") is not True
        or summary.get("information_boundary") != "privileged_selected_age_evaluation_only"
        or type(summary.get("oracle_age_offset")) is not int or summary["oracle_age_offset"] != offset
        or set(summary["evaluations"]) != set(scenarios)
        or runtime != reference["runtime"]
    ):
        raise ValueError(f"P2b-1 run label, scope, or runtime mismatch for {name}")
    results, returns = {}, {}
    for scene in scenarios:
        row = summary["evaluations"][scene]
        if set(row) - {METHOD, "wall_seconds"}:
            raise ValueError(f"unexpected secondary method for {name}/{scene}")
        result = row[METHOD]
        returns[scene] = result_arrays(result, scene, expected_episode_seeds(frozen), METHOD)
        validate_diagnostics(result, offset)
        results[scene] = result
    return {
        "seed": seed, "offset": offset, "sha256": digest, "config_sha256": item["config_sha256"],
        "source_tree_sha256": summary["local_source_provenance"]["tree_sha256"],
        "returns": returns, "results": results,
    }


def compare_preflight(candidate: dict, reference: dict) -> dict:
    actual = candidate["results"][PREFLIGHT_SCENARIO]
    expected = reference["results"][PREFLIGHT_SCENARIO]
    matched = {
        "episode_seeds_equal": actual["episode_seeds"] == expected["episode_seeds"],
        "episode_lengths_equal": actual["episode_lengths"] == expected["episode_lengths"],
        "returns_bitwise_equal": candidate["returns"][PREFLIGHT_SCENARIO].tobytes()
        == np.asarray(expected["returns"], dtype="<f8").tobytes(),
    }
    if not all(matched.values()):
        failed = ", ".join(key for key, ok in matched.items() if not ok)
        raise ValueError(f"P2b-1 zero-offset preflight did not match the archived oracle: {failed}")
    return {
        "status": "passed", "training_seed": SEEDS[0], "scenario": PREFLIGHT_SCENARIO,
        "offset": 0, "episodes": EPISODES, **matched,
        "summary_sha256": candidate["sha256"], "reference_summary_sha256": reference["summary_sha256"],
    }


def verify_preflight(manifest: Path, outputs: Path, reference: Path) -> dict:
    plan, _ = load_plan(manifest)
    zero_reference = load_references(reference)[SEEDS[0]]
    candidate = load_run(plan["preflight"], manifest, outputs, zero_reference)
    return compare_preflight(candidate, zero_reference)


def verify_receipt(receipt: dict, manifest_hash: str, reference: dict, runs: dict, preflight: dict) -> dict:
    identity = receipt["identity"]
    expected_summaries = {
        run_id: {"summary_sha256": run["sha256"], "episodes": len(run["results"]) * EPISODES}
        for run_id, run in runs.items()
    }
    if (
        receipt.get("schema_version") != 1 or receipt["phase"] != "complete"
        or identity["manifest_sha256"] != manifest_hash
        or identity["p2a_plan_sha256"] != P2A_PLAN_SHA256
        or receipt["preflight"] != preflight or receipt["run_summaries"] != expected_summaries
        or any(identity["gpu"][key] != reference["gpu"][key] for key in GPU_FIELDS)
        or identity["package_versions"] != reference["package_versions"]
        or not re.fullmatch(r"[0-9a-f]{40}", identity["git_commit"])
        or not re.fullmatch(r"[0-9a-f]{64}", identity["source_tree_sha256"])
        or any(run["source_tree_sha256"] != identity["source_tree_sha256"] for run in runs.values())
    ):
        raise ValueError("P2b-1 execution receipt does not match the verified runs or P2a runtime")
    return identity


def summarize(samples: np.ndarray) -> dict:
    mean, draws = means_and_bootstrap(samples)
    low, high = interval(draws)
    return {
        "mean": float(mean), "ci95_low": low, "ci95_high": high,
        **{f"seed{seed}_mean": float(values.mean()) for seed, values in zip(SEEDS, samples, strict=True)},
    }


def summarize_diagnostics(results: list[dict]) -> dict:
    fields = ("decision_steps", "effective_offset_sum", "fully_clipped_steps")
    stats = {field: sum(row["age_offset_diagnostics"][field] for row in results) for field in fields}
    return {
        **stats,
        "mean_applied_offset": stats["effective_offset_sum"] / stats["decision_steps"],
        "fully_clipped_fraction": stats["fully_clipped_steps"] / stats["decision_steps"],
    }


def tables_for(references: dict, runs: dict) -> tuple[list[dict], list[dict], list[dict]]:
    absolute, pairs, diagnostics = [], [], []
    for scene in SCENARIOS:
        scores = {
            0: np.asarray([references[seed]["results"][scene]["returns"] for seed in SEEDS], dtype="<f8"),
            **{
                offset: np.stack([runs[seed, offset]["returns"][scene] for seed in SEEDS])
                for offset in OFFSETS
            },
        }
        common = {
            "scenario": scene, "scenario_role": "decision" if scene in DECISION_SCENARIOS else "descriptive",
            "training_seeds": "101 102", "episodes_per_seed": EPISODES,
        }
        for offset, samples in sorted(scores.items()):
            absolute.append({
                **common, "offset": offset, "archived_reference": offset == 0, **summarize(samples),
            })
            if offset != 0:
                diagnostics.append({
                    **common, "offset": offset,
                    **summarize_diagnostics([runs[seed, offset]["results"][scene] for seed in SEEDS]),
                })
        for left, right in ((-1, 1), (-2, 2), *((offset, 0) for offset in OFFSETS)):
            pairs.append({
                **common, "offset_left": left, "offset_right": right,
                **summarize(scores[left] - scores[right]),
            })
    return absolute, pairs, diagnostics


def decision_for(pairs: list[dict]) -> dict:
    gates = {}
    for magnitude in (1, 2):
        rows = {
            row["scenario"]: row for row in pairs
            if row["offset_left"] == -magnitude and row["offset_right"] == magnitude
            and row["scenario"] in DECISION_SCENARIOS
        }
        if set(rows) != set(DECISION_SCENARIOS):
            raise ValueError(f"incomplete P2b-1 decision interval grid for ±{magnitude}")
        negatives = [scene for scene in DECISION_SCENARIOS if rows[scene]["ci95_high"] < 0]
        gates[f"minus{magnitude}_vs_plus{magnitude}"] = {
            "negative_scenarios": negatives, "passes": len(negatives) >= 3,
        }
    return {
        "stage": "p2b1", "decision_scenarios": list(DECISION_SCENARIOS),
        "descriptive_scenarios": list(DESCRIPTIVE_SCENARIOS), "offset_pairs": gates,
        "p2b1_prediction_passed": all(gate["passes"] for gate in gates.values()),
        "negative_interval_definition": "upper 95% confidence bound strictly below zero",
        "minimum_negative_scenarios_per_pair": 3,
    }


def report_text(pairs: list[dict], decision: dict) -> str:
    lines = [
        "# P2b-1 Hopper age-offset diagnostic", "",
        f"The analysis paired {NEW_EPISODES} new privileged offset episodes with {REFERENCE_EPISODES} "
        "archived oracle-zero reference episodes across two training seeds. "
        "A separate same-session check reproduced all 20 archived zero-offset episodes bitwise.", "",
        "The endpoint is raw Hopper return, with equal training-seed weights and paired 95% bootstrap "
        "intervals. The table shows negative offset minus positive offset. IID is descriptive.", "",
        "| Scenario | Role | −1 vs +1 | −2 vs +2 |", "| --- | --- | ---: | ---: |",
    ]
    for scene in SCENARIOS:
        fields = []
        for magnitude in (1, 2):
            row = next(
                row for row in pairs if row["scenario"] == scene
                and (row["offset_left"], row["offset_right"]) == (-magnitude, magnitude)
            )
            fields.append(f"{row['mean']:.3f} [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]")
        role = "Decision" if scene in DECISION_SCENARIOS else "Descriptive"
        lines.append(f"| {scene} | {role} | {' | '.join(fields)} |")
    lines.extend([
        "", f"Preregistered prediction passed **{decision['p2b1_prediction_passed']}**. "
        "Both offset pairs must have a negative interval in at least three of the four decision scenarios.",
        "", "The paired CSV records each training seed's return difference. The offset diagnostics "
        "report realized shifts and complete clipping rates pooled over decision steps. "
        "The two seeds support this diagnostic and leave independent generalization untested.", "",
    ])
    return "\n".join(lines)


def aggregate(manifest: Path, outputs: Path, reference: Path, output: Path) -> dict:
    plan, manifest_hash = load_plan(manifest)
    references = load_references(reference)
    receipt, receipt_hash = load_json(outputs / "execution_receipt.json")
    descriptors = [plan["preflight"], *plan["runs"]]
    if set(receipt["run_summaries"]) != {item["run_id"] for item in descriptors}:
        raise ValueError("P2b-1 execution receipt is missing or adds run summaries")
    runs = {
        item["run_id"]: load_run(
            item, manifest, outputs, references[item["seed"]],
            receipt["run_summaries"][item["run_id"]]["summary_sha256"],
        )
        for item in descriptors
    }
    zero = runs[plan["preflight"]["run_id"]]
    preflight = compare_preflight(zero, references[101])
    identity = verify_receipt(receipt, manifest_hash, references[101], runs, preflight)
    absolute, pairs, diagnostics = tables_for(
        references, {(run["seed"], run["offset"]): run for run in runs.values() if run["offset"] != 0},
    )
    decision = decision_for(pairs)
    evidence = {
        "schema_version": 1, "stage": "p2b1", "manifest_sha256": manifest_hash,
        "execution_receipt_sha256": receipt_hash, "p2a_plan_sha256": P2A_PLAN_SHA256,
        "p2a_evidence_sha256": P2A_EVIDENCE_SHA256, "gpu_model": identity["gpu"]["name"],
        "gpu_driver": identity["gpu"]["driver_version"],
        "package_versions": identity["package_versions"],
        "git_commit": identity["git_commit"], "source_tree_sha256": identity["source_tree_sha256"],
        "new_episodes": NEW_EPISODES, "reference_episodes": REFERENCE_EPISODES, "preflight": preflight,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "resamples": RESAMPLES, "unit": "training_seed_then_episode_pair"},
        "references": [
            {"training_seed": seed, **{key: data[key] for key in ("config_sha256", "receipt_sha256", "summary_sha256")}}
            for seed, data in references.items()
        ],
        "runs": [
            {key: run[key] for key in ("seed", "offset", "sha256", "config_sha256")} for run in runs.values()
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("absolute.csv", absolute), ("paired.csv", pairs), ("offset_diagnostics.csv", diagnostics)):
        write_csv(output / name, rows)
    write_json(output / "manifest.json", evidence)
    write_json(output / "decision.json", decision)
    (output / "report.md").write_text(report_text(pairs, decision), encoding="utf-8")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("manifest", "outputs", "reference", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    decision = aggregate(args.manifest, args.outputs, args.reference, args.output)
    print(json.dumps({"p2b1_prediction_passed": decision["p2b1_prediction_passed"]}))


if __name__ == "__main__":
    main()
