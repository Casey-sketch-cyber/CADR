"""Analyze verified P2a work units with paired, seed-first bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
from functools import cache
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from prepare_t6 import CLEAN_EXCEPTIONS, CORE, HOPPER_EXTRA, TASK_SEEDS


BOOTSTRAP_SEED = 20261124
RESAMPLES = 10_000
EPISODES = 20
CLEAN = "clean_delay_0"
PRIMARY = {"hopper": "raw_return", "pickcube": "success_once"}
DP_ENDPOINTS = {
    "success_once": "successes", "success_at_end": "successes_at_end",
    "success_occupancy": "success_occupancies",
    "post_change_success_occupancy": "post_change_success_occupancies",
}
SCENARIOS = {
    "hopper": (
        CLEAN, "fixed_delay_1", "fixed_delay_2", "fixed_delay_4", "fixed_delay_8",
        "fixed_delay_16", "iid_in_support", "markov_switching", "changepoint_in_support",
        "drifting_to_long_delay", "ood_support_shift", "ood_changepoint",
    ),
    "pickcube": (
        CLEAN, "fixed_delay_1", "fixed_delay_2", "fixed_delay_4", "fixed_delay_8",
        "iid_in_distribution", "markov_switching", "ood_changepoint", "ood_drifting",
    ),
}
PRIORITY = ("drifting_to_long_delay", "markov_switching", "ood_support_shift", "ood_changepoint")
CORE_CONTRASTS = {
    "hazard_vs_legacy": ("full_hazard", "full_legacy"),
    "hazard_vs_in_order": ("full_hazard", "in_order_age"),
    "censoring": ("full_hazard", "no_censoring"),
    "oracle_vs_hazard": ("cadr_oracle_age", "full_hazard"),
    "oracle_vs_direct": ("cadr_oracle_age", "direct"),
    "clean_direct_vs_oracle": ("clean_direct", "cadr_oracle_age"),
}


def load_json(path: Path, expected_hash: str | None = None) -> tuple[dict, str]:
    content = path.read_bytes()
    digest = sha256(content).hexdigest()
    if expected_hash is not None and digest != expected_hash:
        raise ValueError(f"SHA-256 mismatch at {path}")
    return json.loads(content), digest


def relative_file(root: Path, name: str) -> Path:
    path = Path(name)
    target = (root / path).resolve()
    if path.is_absolute() or ".." in path.parts or not target.is_relative_to(root):
        raise ValueError(f"unsafe relative filename {name!r}")
    return target


def validate_plan(plan: dict) -> dict:
    if plan.get("schema_version") != 1 or plan.get("stage") != "p2a":
        raise ValueError("expected the reviewed P2a manifest")
    batches = {}
    for batch in plan["batches"]:
        task, seed = batch["task"], batch["training_seed"]
        expected = CORE + HOPPER_EXTRA if task == "hopper" and seed in (101, 102) else CORE
        ids = [item["method_id"] for item in batch["runs"]]
        if (
            seed not in TASK_SEEDS.get(task, ()) or (task, seed) in batches
            or batch["batch_id"] != f"{task}_seed{seed}"
            or tuple(batch["scenarios"]) != SCENARIOS[task] or batch["eval_episodes"] != EPISODES
            or len(ids) != len(expected) or set(ids) != set(expected)
            or batch["expected_episodes"] != len(expected) * len(SCENARIOS[task]) * EPISODES
            or any(item["clean_expected_equal"] != (item["method_id"] not in CLEAN_EXCEPTIONS)
                   for item in batch["runs"])
        ):
            raise ValueError(f"unexpected or duplicate P2a batch {task}/{seed}")
        batches[task, seed] = batch
    if (
        set(batches) != {(task, seed) for task, seeds in TASK_SEEDS.items() for seed in seeds}
        or plan["method_runs"] != 58 or plan["expected_episodes"] != 13200
    ):
        raise ValueError("P2a manifest must plan all 58 method work units and 13,200 episodes")
    return batches


def numeric_vector(result: dict, field: str, *, rate: bool = False) -> np.ndarray:
    values = np.asarray(result[field], dtype=np.float64)
    if values.shape != (EPISODES,) or not np.isfinite(values).all():
        raise ValueError(f"expected {EPISODES} finite values for {field}")
    if rate and (np.any(values < 0) or np.any(values > 1)):
        raise ValueError(f"out-of-range probability for {field}")
    return values


def validate_result(task: str, scenario: dict, expected_seeds: list[int], expected_method: str,
                    is_oracle: bool, result: dict) -> dict[str, np.ndarray]:
    if (
        result["episode_seeds"] != expected_seeds or result["episodes"] != EPISODES
        or result["scenario"] != scenario["name"] or result["method"] != expected_method
        or result["privileged_selected_age"] is not is_oracle
        or len(result["episode_lengths"]) != EPISODES
    ):
        raise ValueError(f"invalid pairing, label or information boundary: {task}/{scenario['name']}")
    returns = numeric_vector(result, "returns")
    if task == "hopper":
        return {"raw_return": returns}
    values = {}
    for endpoint, field in DP_ENDPOINTS.items():
        if endpoint == "post_change_success_occupancy":
            if scenario["type"] != "changepoint":
                continue
            if len(result.get(field, [])) != EPISODES:
                # The legacy evaluator omits episodes ending before the change. An
                # incomplete array cannot be treated as a fully paired endpoint.
                continue
        vector = numeric_vector(result, field, rate=True)
        if endpoint in {"success_once", "success_at_end"} and not np.isin(vector, [0, 1]).all():
            raise ValueError(f"nonbinary data in {field}")
        values[endpoint] = 100.0 * vector
    return values


def timing_row(task: str, seed: int, gpu: str, method: str, scenario: dict, summary: dict,
               config: dict, result: dict) -> dict:
    timing = result["controller_timing"]
    context = summary["controller_timing_context"]
    p50, p95 = result["inference_ms_p50"], result["inference_ms_p95"]
    if (
        timing["scope"] != "controller.act" or timing["cuda_synchronized"] is not True
        or timing["samples"] != sum(result["episode_lengths"])
        or not np.isfinite([p50, p95]).all() or not 0 <= p50 <= p95
        or context["max_delay"] != config["max_delay"]
    ):
        raise ValueError(f"invalid synchronized timing: {task}/{seed}/{method}/{scenario['name']}")
    chunk = config.get("act_horizon", 8) if method == "direct" else config.get(
        "cadr_act_horizon", config.get("act_horizon", 8)
    )
    fallback = result.get("hazard_zero_mass_fallback")
    return {
        "task": task, "training_seed": seed, "gpu_model": gpu, "scenario": scenario["name"],
        "method": method, "scope": timing["scope"], "cuda_synchronized": True,
        "samples": timing["samples"], "p50_ms": p50, "p95_ms": p95,
        "max_delay": context["max_delay"],
        "dynamics_members": 0 if method == "direct" else context["cadr_ensemble_size"],
        "action_chunk": chunk if task == "pickcube" else 1,
        "hazard_zero_mass_fallback_episode_mean": fallback,
    }


def clean_comparison(task: str, direct: dict, result: dict) -> dict[str, bool]:
    fields = ("returns", "successes", "successes_at_end", "success_occupancies")
    equal = {field: direct[field] == result[field] for field in ("episode_seeds", "episode_lengths")}
    for field in fields if task == "pickcube" else fields[:1]:
        left, right = (np.asarray(record[field], dtype="<f8").tobytes() for record in (direct, result))
        equal[field] = left == right
    return equal


def load_batch(batch: dict, manifest_path: Path, manifest_hash: str, outputs: Path) -> dict:
    task, seed, batch_id = (batch[key] for key in ("task", "training_seed", "batch_id"))
    evidence = {"task": task, "training_seed": seed, "batch_id": batch_id}
    root = relative_file(outputs, batch_id)
    receipt_path = relative_file(root, "batch_receipt.json")
    if not receipt_path.is_file():
        return {"evidence": {**evidence, "status": "missing"}}
    receipt, receipt_hash = load_json(receipt_path)
    identity = receipt["identity"]
    if (
        receipt.get("schema_version") != 1 or identity["manifest_sha256"] != manifest_hash
        or any(identity[key] != value for key, value in evidence.items())
        or receipt["run_plan"] != batch["runs"] or receipt["inputs"] != batch["inputs"]
        or receipt["scenarios"] != batch["scenarios"] or receipt["eval_episodes"] != EPISODES
    ):
        raise ValueError(f"receipt identity or configuration mismatch for {batch_id}")
    gpu = identity["gpu"]["name"]
    phase = receipt["phase"]
    if phase not in {"complete", "failed", "running"}:
        raise ValueError(f"unknown phase for {batch_id}")
    evidence.update(status=phase, gpu_model=gpu, receipt_sha256=receipt_hash)
    if phase != "complete":
        return {"evidence": evidence}
    methods = {item["method_id"] for item in batch["runs"]}
    if set(receipt["runs"]) != methods or set(receipt["clean_checks"]) != methods:
        raise ValueError(f"completed receipt is missing or adds methods in {batch_id}")

    metrics, timing, clean_results, method_evidence = {}, [], {}, {}
    runtimes, expected_seed_keys, missing_post_change = [], [], []
    for item in batch["runs"]:
        method, key = item["method_id"], item["evaluation_key"]
        record = receipt["runs"][method]
        if record["phase"] != "complete":
            raise ValueError(f"incomplete method in a completed batch: {batch_id}/{method}")
        config, _ = load_json(relative_file(manifest_path.parent, item["config"]), item["config_sha256"])
        expected_seeds = [config["eval_seed"] + 100_003 * episode for episode in range(EPISODES)]
        expected_seed_keys.append(expected_seeds)
        summary, digest = load_json(relative_file(root, record["summary"]), record["summary_sha256"])
        if (
            summary["deployment_parameter_updates"] is not False or config["seed"] != seed
            or summary["evaluation_methods"] != [key] or config["evaluation_methods"] != [key]
            or tuple(scene["name"] for scene in config["scenarios"]) != SCENARIOS[task]
            or set(summary["evaluations"]) != set(batch["scenarios"])
            or summary["local_source_provenance"]["tree_sha256"] != identity["source_tree_sha256"]
        ):
            raise ValueError(f"summary provenance or configuration mismatch for {batch_id}/{method}")
        runtime = summary["runtime"]
        runtimes.append({name: runtime[name] for name in ("device", "cuda", "torch")})
        for scenario in config["scenarios"]:
            name = scenario["name"]
            scenario_results = summary["evaluations"][name]
            if set(scenario_results) - {key, "wall_seconds"}:
                raise ValueError(f"unexpected secondary method in {batch_id}/{method}/{name}")
            result = scenario_results[key]
            metrics[method, name] = validate_result(
                task, scenario, expected_seeds, config["required_methods"][0],
                method == "cadr_oracle_age", result,
            )
            if task == "pickcube" and scenario["type"] == "changepoint" and (
                "post_change_success_occupancy" not in metrics[method, name]
            ):
                missing_post_change.append({"method": method, "scenario": name,
                                            "reported_episodes": len(result.get("post_change_success_occupancies", []))})
            timing.append(timing_row(task, seed, gpu, method, scenario, summary, config, result))
            if name == CLEAN:
                clean_results[method] = result
        method_evidence[method] = {
            "summary_sha256": digest, "config_sha256": item["config_sha256"],
            "evaluation_key": key, "evaluator_method": config["required_methods"][0],
            **{field: record[field] for field in ("attempt_count", "resume_count", "native_crash_count")},
        }
    if any(runtime != runtimes[0] for runtime in runtimes) or any(
        keys != expected_seed_keys[0] for keys in expected_seed_keys
    ):
        raise ValueError(f"different evaluation seeds or runtime across methods in {batch_id}")
    torch_device = runtimes[0]["device"]
    if torch_device.lower().removeprefix("nvidia ") != gpu.lower().removeprefix("nvidia "):
        raise ValueError(f"reported Torch device does not match the batch GPU in {batch_id}")

    clean_checks = []
    for method, result in clean_results.items():
        expected = method not in CLEAN_EXCEPTIONS
        fields = clean_comparison(task, clean_results["direct"], result)
        equal = all(fields.values())
        if receipt["clean_checks"][method] != {
            "expected_equal": expected, "bitwise_equal": equal, "field_equal": fields,
        } or (expected and not equal):
            raise ValueError(f"mandatory clean correspondence failed in {batch_id}/{method}")
        clean_checks.append({
            "task": task, "training_seed": seed, "gpu_model": gpu, "method": method,
            "expected_equal": expected, "bitwise_equal": equal,
            "unequal_fields": " ".join(field for field, matched in fields.items() if not matched),
            "exception": CLEAN_EXCEPTIONS.get(method),
        })
    runtime_evidence = {
        "gpu_model": gpu, "driver": identity["gpu"].get("driver_version"),
        "gpu_memory_mib": identity["gpu"].get("memory.total"),
        "source_tree_sha256": identity["source_tree_sha256"],
        "package_versions": identity.get("package_versions", {}), **runtimes[0],
    }
    stratum = sha256(json.dumps(runtime_evidence, sort_keys=True).encode()).hexdigest()[:12]
    evidence.update(runtime_evidence, stratum=stratum, methods=method_evidence,
                    input_digests={item["role"]: item["sha256"] for item in batch["inputs"]})
    if missing_post_change:
        evidence["incomplete_post_change_endpoint"] = missing_post_change
    return {"evidence": evidence, "metrics": metrics, "timing": timing, "clean_checks": clean_checks}


@cache
def bootstrap_indices(n_seeds: int, n_episodes: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    seed_indices = rng.integers(n_seeds, size=(RESAMPLES, n_seeds, 1))
    episode_indices = rng.integers(n_episodes, size=(RESAMPLES, n_seeds, n_episodes))
    return seed_indices, episode_indices


def means_and_bootstrap(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    means = samples.mean(axis=(0, 1))
    if len(samples) < 2:
        return means, None
    indices = bootstrap_indices(*samples.shape[:2])
    return means, samples[indices].mean(axis=(1, 2))


def interval(draws: np.ndarray | None) -> tuple[float | None, float | None]:
    return tuple(map(float, np.percentile(draws, (2.5, 97.5)))) if draws is not None else (None, None)


def recovery(means: np.ndarray, draws: np.ndarray | None, scenario: str) -> dict:
    method, direct, clean = means
    denominator = clean - direct
    status = "clean_reference_itself" if scenario == CLEAN else "nonpositive_denominator"
    result = {"recovery_pct": None, "recovery_ci95_low": None, "recovery_ci95_high": None,
              "recovery_positive_denominator_fraction": None, "recovery_status": status}
    if scenario == CLEAN:
        return result
    denominators = draws[:, 2] - draws[:, 1] if draws is not None else None
    if denominators is not None:
        result["recovery_positive_denominator_fraction"] = float(np.mean(denominators > 0))
    if denominator <= 0:
        return result
    result["recovery_pct"] = 100.0 * (method - direct) / denominator
    if denominators is None:
        result["recovery_status"] = "single_seed_descriptive"
    elif np.any(denominators <= 0):
        result["recovery_status"] = "bootstrap_denominator_unstable"
    else:
        result["recovery_status"] = "ok"
        low, high = interval(100.0 * (draws[:, 0] - draws[:, 1]) / denominators)
        result.update(recovery_ci95_low=low, recovery_ci95_high=high)
    return result


def expected_seeds(task: str, *methods: str) -> tuple[int, ...]:
    return (101, 102) if any(method in HOPPER_EXTRA for method in methods) else TASK_SEEDS[task]


def cohorts(batches: dict, task: str, seeds: tuple[int, ...], fields: tuple) -> list[tuple[dict, np.ndarray]]:
    by_stratum: dict[str, dict[int, np.ndarray]] = {}
    gpu_names = {}
    for seed in seeds:
        batch = batches[task, seed]
        metrics = batch.get("metrics", {})
        columns = [metrics.get((method, scenario), {}).get(endpoint)
                   for method, scenario, endpoint in fields]
        if any(value is None for value in columns):
            continue
        evidence = batch["evidence"]
        stratum = evidence["stratum"]
        gpu_names[stratum] = evidence["gpu_model"]
        by_stratum.setdefault(stratum, {})[seed] = np.column_stack(columns)
    groups = []
    for stratum, group in sorted(by_stratum.items()):
        included = tuple(group)
        scope = "complete_same_gpu" if included == seeds else "incomplete_gpu_stratum"
        metadata = {
            "task": task, "gpu_model": gpu_names[stratum], "stratum": stratum, "scope": scope,
            "training_seeds": " ".join(map(str, included)), "seed_count": len(included),
            "expected_training_seeds": " ".join(map(str, seeds)), "episodes_per_seed": EPISODES,
        }
        groups.append((metadata, np.stack(list(group.values()))))
    return groups


def contrast_plan(task: str) -> dict[str, tuple[str, str]]:
    pairs = dict(CORE_CONTRASTS)
    methods = CORE + HOPPER_EXTRA if task == "hopper" else CORE
    for method in methods:
        if method not in {"direct", "cadr_oracle_age"}:
            pairs[f"{method}_vs_direct"] = (method, "direct")
        if method in HOPPER_EXTRA:
            pairs[f"hazard_vs_{method}"] = ("full_hazard", method)
    return pairs


def endpoint_list(task: str, scenario: str) -> tuple[str, ...]:
    if task == "hopper":
        return (PRIMARY[task],)
    if scenario == "ood_changepoint":
        return tuple(DP_ENDPOINTS)
    return tuple(DP_ENDPOINTS)[:-1]


def build_tables(batches: dict, task: str) -> tuple[list[dict], list[dict]]:
    absolute, paired = [], []
    methods = CORE + HOPPER_EXTRA if task == "hopper" else CORE
    units = "unscaled_return" if task == "hopper" else "percent"
    for scenario in SCENARIOS[task]:
        for endpoint in endpoint_list(task, scenario):
            for method in methods:
                fields = ((method, scenario, endpoint), ("direct", scenario, endpoint))
                if endpoint == PRIMARY[task]:
                    fields += (("direct", CLEAN, endpoint),)
                for metadata, samples in cohorts(batches, task, expected_seeds(task, method), fields):
                    means, draws = means_and_bootstrap(samples)
                    low, high = interval(draws[:, 0] if draws is not None else None)
                    row = {
                        **metadata, "scenario": scenario, "endpoint": endpoint, "units": units,
                        "method": method, "privileged": method == "cadr_oracle_age",
                        "score": float(means[0]), "ci95_low": low, "ci95_high": high,
                        "direct_score": float(means[1]),
                        "same_environment_normalized_score": None,
                    }
                    if endpoint == PRIMARY[task]:
                        row.update(clean_direct_score=float(means[2]), **recovery(means, draws, scenario))
                    absolute.append(row)
            for name, (method_a, method_b) in contrast_plan(task).items():
                if method_a == "clean_direct" and endpoint != PRIMARY[task]:
                    continue
                real_method_a = "direct" if method_a == "clean_direct" else method_a
                scene_a = CLEAN if method_a == "clean_direct" else scenario
                fields = ((real_method_a, scene_a, endpoint), (method_b, scenario, endpoint))
                for metadata, samples in cohorts(batches, task, expected_seeds(task, method_a, method_b), fields):
                    by_episode = samples[:, :, 0] - samples[:, :, 1]
                    delta, draws = means_and_bootstrap(by_episode)
                    low, high = interval(draws)
                    direction = interval_direction(low, high)
                    seed_deltas = by_episode.mean(axis=1)
                    seeds = metadata["training_seeds"].split()
                    paired.append({
                        **metadata, "scenario": scenario, "endpoint": endpoint, "contrast": name,
                        "method_a": method_a, "method_b": method_b, "score_a": float(samples[:, :, 0].mean()),
                        "score_b": float(samples[:, :, 1].mean()), "units": units if task == "hopper" else "percentage_points",
                        "delta": float(delta), "ci95_low": low, "ci95_high": high, "interval_direction": direction,
                        "seed_direction": "positive" if np.all(seed_deltas > 0) else (
                            "negative" if np.all(seed_deltas < 0) else "mixed_or_zero"
                        ),
                        **{f"seed{seed}_delta": float(value) for seed, value in zip(seeds, seed_deltas)},
                    })
    return absolute, paired


def interval_direction(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "not_estimated"
    if low > 0:
        return "positive"
    if high < 0:
        return "negative"
    return "includes_zero"


def assess_decision(hopper: list[dict], pickcube: list[dict]) -> dict:
    answer = {
        "basis": "all five Hopper training seeds on one GPU model and runtime",
        "priority_scenarios": PRIORITY,
        "pickcube_role": "two-seed direction check only",
        "interval_criteria": "positive if lower bound > 0; clearly negative if upper bound < 0",
        "route_b_interpretation": "intervals including zero indicate no discernible difference, not equivalence",
        "oracle_interpretation": "descriptive size and direction; no threshold for a large gap was specified",
    }
    authoritative = {
        (row["contrast"], row["scenario"]): row for row in hopper
        if row["scope"] == "complete_same_gpu" and row["endpoint"] == PRIMARY["hopper"]
        and row["seed_count"] == 5 and row["contrast"] in CORE_CONTRASTS
    }
    answer["pickcube_direction"] = {
        scenario: {row["contrast"]: row["seed_direction"] for row in pickcube
                   if row["scenario"] == scenario and row["endpoint"] == PRIMARY["pickcube"]
                   and row["scope"] == "complete_same_gpu" and row["contrast"] in CORE_CONTRASTS}
        for scenario in SCENARIOS["pickcube"]
    }
    if set(authoritative) != {(name, scene) for name in CORE_CONTRASTS for scene in SCENARIOS["hopper"]}:
        return {**answer, "status": "insufficient_complete_homogeneous_hopper_data"}

    def direction(contrast: str, scenario: str) -> str:
        return authoritative[contrast, scenario]["interval_direction"]

    # Route A checks in-order age in the four priority scenarios and checks
    # the legacy posterior in every Hopper scenario.
    positive_in_order = [scene for scene in PRIORITY if direction("hazard_vs_in_order", scene) == "positive"]
    negative_in_order = [scene for scene in PRIORITY if direction("hazard_vs_in_order", scene) == "negative"]
    negative_legacy = [scene for scene in SCENARIOS["hopper"] if direction("hazard_vs_legacy", scene) == "negative"]
    # The long-delay drift and both OOD cases are explicit in the design. Also
    # display all censoring estimates so the review can inspect fixed/Markov cases.
    long_delay = ("drifting_to_long_delay", "ood_support_shift", "ood_changepoint")
    positive_censoring = [scene for scene in long_delay if direction("censoring", scene) == "positive"]
    return {
        **answer, "status": "reviewable", "route_a_criteria_met": (
            len(positive_in_order) >= 2 and not negative_in_order and not negative_legacy
        ),
        "priority_positive_hazard_vs_in_order": positive_in_order,
        "priority_clearly_negative_hazard_vs_in_order": negative_in_order,
        "all_clearly_negative_hazard_vs_legacy": negative_legacy,
        "long_delay_censoring_scenarios": long_delay,
        "positive_long_delay_censoring": positive_censoring,
        "censoring_title_criterion_met": bool(positive_censoring),
        "route_b_discussion_flag": all(direction("hazard_vs_in_order", scene) == "includes_zero"
                                       for scene in SCENARIOS["hopper"]),
        "hopper_priority_contrasts": {
            scene: {name: {key: row[key] for key in ("delta", "ci95_low", "ci95_high", "interval_direction")}
                    for name in CORE_CONTRASTS for row in (authoritative[name, scene],)}
            for scene in PRIORITY
        },
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def report_text(evidence: list[dict], decision: dict, tables: dict) -> str:
    lines = [
        "# CADR T6 P2a paired evaluation", "",
        "P2a compares frozen controllers. The manifest lists completed batches, source and input hashes, and retry counts.",
        "", "| Task | Training seed | Receipt | GPU |", "| --- | ---: | --- | --- |",
    ]
    lines.extend(f"| {row['task']} | {row['training_seed']} | {row['status']} | {row.get('gpu_model', 'NA')} |"
                 for row in evidence)
    lines.extend([
        "", "The absolute tables show unscaled Hopper return and PickCube success percentages. The paired tables include",
        "every seed-specific difference, secondary PickCube endpoints and the three complementary oracle comparisons.",
        "Hopper core comparisons use five training seeds. Hopper extras and PickCube use two. Each seed has 20 paired episodes.",
        "Equal-weight seed means and 95% percentile intervals use 10,000 seed-first resamples with seed 20261124.",
        "Incomplete or incompatible cohorts appear as separate strata. Single-seed strata have descriptive values only.",
        "The oracle receives true age during evaluation. Inspect negative oracle gaps as recorded.",
        "", "The clean gate compares saved return, length and applicable success arrays bitwise. The documented exceptions",
        "are listed in `clean_checks.csv`. The evaluator's separate entry-point equivalence check covers actions.",
        "`timing.csv` gives per-method, scenario and training seed synchronized CUDA `controller.act` p50/p95;",
        "`fixed_prior_matched` uses a static reference distribution and its fallback rate is in that file.",
        "", "`historical_d4rl_hopper.csv` rescales raw return using the old D4RL reference values −20.272305 and 3234.3.",
        "Hopper-v5 same-environment normalization remains NA. Empty CSV cells denote unavailable estimates.",
        "", f"## Decision status — {decision['status']}", "",
    ])
    if decision["status"] == "reviewable":
        lines.extend([
            f"- Route A criteria met: {decision['route_a_criteria_met']}.",
            f"- Long-delay censoring title criterion met: {decision['censoring_title_criterion_met']}.",
            f"- Route B discussion flag: {decision['route_b_discussion_flag']}.",
            "- Oracle gaps are descriptive. The design did not specify a numerical threshold for a large gap.",
            "", "The following five-seed Hopper contrasts use raw return units.", "",
            "| Priority scenario | Hazard − legacy | Hazard − in-order | Hazard − uncensored | Oracle − hazard |",
            "| --- | ---: | ---: | ---: | ---: |",
        ])
        for scene, contrasts in decision["hopper_priority_contrasts"].items():
            cols = [f"{row['delta']:.3f} [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]"
                    for name in ("hazard_vs_legacy", "hazard_vs_in_order", "censoring", "oracle_vs_hazard")
                    for row in (contrasts[name],)]
            lines.append(f"| {scene} | {' | '.join(cols)} |")
    else:
        lines.append("A pooled decision requires all five Hopper seeds on the same GPU model and runtime.")
    lines.extend(["", "PickCube supports directional comparisons only. See `decision.json` and the full paired tables."])
    for task in SCENARIOS:
        absolute, paired = tables[task]
        if task == "pickcube" and not any(row["endpoint"] == "post_change_success_occupancy"
                                         and row["scope"] == "complete_same_gpu" for row in absolute):
            lines.append("PickCube has no complete pooled post-change occupancy cohort in these receipts.")
        lines.append(f"- {task}: {len(absolute)} absolute rows and {len(paired)} paired rows.")
    return "\n".join(lines) + "\n"


def aggregate(manifest: Path, outputs: Path, output: Path) -> dict:
    manifest, outputs, output = (path.resolve() for path in (manifest, outputs, output))
    if output.exists():
        raise FileExistsError(output)
    plan, plan_hash = load_json(manifest)
    batches = {key: load_batch(batch, manifest, plan_hash, outputs) for key, batch in validate_plan(plan).items()}
    tables = {task: build_tables(batches, task) for task in SCENARIOS}
    decision = assess_decision(tables["hopper"][1], tables["pickcube"][1])
    evidence = [batch["evidence"] for batch in batches.values()]
    output.mkdir(parents=True, exist_ok=False)
    for task, (absolute, paired) in tables.items():
        write_csv(output / f"absolute_{task}.csv", absolute)
        write_csv(output / f"paired_{task}.csv", paired)
    historical = [
        {key: row[key] for key in ("task", "gpu_model", "stratum", "scope", "training_seeds", "scenario", "method")}
        | {f"old_d4rl_scale_{field}": None if row[field] is None else 100 * (row[field] + 20.272305) / (3234.3 + 20.272305)
           for field in ("score", "ci95_low", "ci95_high")}
        for row in tables["hopper"][0]
    ]
    write_csv(output / "historical_d4rl_hopper.csv", historical)
    for key, filename in (("timing", "timing.csv"), ("clean_checks", "clean_checks.csv")):
        write_csv(output / filename, [row for batch in batches.values() for row in batch.get(key, [])])
    write_json(output / "decision.json", decision)
    report = {
        "schema_version": 1, "stage": "p2a", "plan_sha256": plan_hash,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "resamples": RESAMPLES,
                      "method": "paired_seed_then_episode_percentile"},
        "expected_episodes": plan["expected_episodes"],
        "verified_episodes": sum(len(batch.get("metrics", {})) * EPISODES for batch in batches.values()),
        "batches": evidence,
    }
    write_json(output / "manifest.json", report)
    (output / "report.md").write_text(report_text(evidence, decision, tables), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("manifest", "outputs", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate(args.manifest, args.outputs, args.output)
    print(json.dumps({key: report[key] for key in ("stage", "verified_episodes", "expected_episodes")}))


if __name__ == "__main__":
    main()
