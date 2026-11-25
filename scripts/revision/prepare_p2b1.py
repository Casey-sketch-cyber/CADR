"""Materialize the reviewed Hopper oracle-age offset evaluation plan."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from prepare_t6 import method_settings, write_config


P2A_PLAN = Path("results/revision/T6/p2a_plan.json")
P2A_PLAN_SHA256 = "2b990129936cff5a052bbbb7d8ed52f8ff9285ed9609c9a7d27440a832d355b2"
SEEDS = (101, 102)
OFFSETS = ((-2, "m2"), (-1, "m1"), (1, "p1"), (2, "p2"))
DECISION_SCENARIOS = ("fixed_delay_4", "fixed_delay_8", "markov_switching", "ood_support_shift")
DESCRIPTIVE_SCENARIOS = ("iid_in_support",)
PREFLIGHT_SCENARIOS = ("fixed_delay_8",)
EVAL_EPISODES = 20
METHOD = "cadr_oracle_age_offset"


def archived_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    # Git stores these JSON files with LF. A Windows checkout can convert them to CRLF.
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    if sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"published P2a input digest differs: {path}")
    return json.loads(raw)


def source_config(project_root: Path, batch: dict[str, Any]) -> dict[str, Any]:
    template = archived_json(project_root / batch["source_config"], batch["source_config_sha256"])
    config, method = method_settings(template, "hopper", "cadr_oracle_age")
    oracle = next(run for run in batch["runs"] if run["method_id"] == method)
    # Match the canonical serializer used by the P2a generator and P2a runner.
    canonical = (json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if sha256(canonical).hexdigest() != oracle["config_sha256"]:
        raise ValueError(f"could not reproduce the archived oracle config for {batch['batch_id']}")
    return config


def write_run(
    output: Path,
    batch: dict[str, Any],
    archived: dict[str, Any],
    offset: int,
    offset_name: str,
    scenario_names: tuple[str, ...],
) -> dict[str, Any]:
    seed = batch["training_seed"]
    run_id = f"hopper_seed{seed}_offset_{offset_name}"
    relative_path = Path("configs") / f"{run_id}.json"
    scenarios = [scene for scene in archived["scenarios"] if scene["name"] in scenario_names]
    names = [scene["name"] for scene in scenarios]
    if len(names) != len(scenario_names) or set(names) != set(scenario_names):
        raise ValueError(f"reviewed scenario is missing or repeated in {batch['batch_id']}")
    config = dict(
        archived,
        scenarios=scenarios,
        experiment_series="ablation_v2_p2b1_hopper",
        evaluation_methods=[METHOD],
        required_methods=[METHOD],
        include_cadr_oracle_age=False,
        oracle_age_offset=offset,
    )
    return {
        "run_id": run_id,
        "seed": seed,
        "offset": offset,
        "config": relative_path.as_posix(),
        "config_sha256": write_config(output / relative_path, config),
        "scenarios": names,
        "inputs": batch["inputs"],
    }


def prepare(project_root: Path, output: Path) -> dict[str, Any]:
    source = archived_json(project_root / P2A_PLAN, P2A_PLAN_SHA256)
    scenario_names = DECISION_SCENARIOS + DESCRIPTIVE_SCENARIOS
    runs = []
    for seed in SEEDS:
        batch = next(row for row in source["batches"]
                     if row["task"] == "hopper" and row["training_seed"] == seed)
        archived = source_config(project_root, batch)
        runs.extend(write_run(output, batch, archived, offset, label, scenario_names)
                    for offset, label in OFFSETS)
        if seed == SEEDS[0]:
            preflight = write_run(output, batch, archived, 0, "0", PREFLIGHT_SCENARIOS)

    manifest = {
        "schema_version": 1,
        "stage": "p2b1",
        "p2a_plan_sha256": P2A_PLAN_SHA256,
        "decision_scenarios": list(DECISION_SCENARIOS),
        "descriptive_scenarios": list(DESCRIPTIVE_SCENARIOS),
        "eval_episodes": EVAL_EPISODES,
        "new_episodes": len(runs) * len(scenario_names) * EVAL_EPISODES,
        "reference_episodes": len(SEEDS) * len(scenario_names) * EVAL_EPISODES,
        "preflight_episodes": len(PREFLIGHT_SCENARIOS) * EVAL_EPISODES,
        "runs": runs,
        "preflight": preflight,
    }
    write_config(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = prepare(args.project_root.resolve(), args.output.resolve())
    print(json.dumps({key: plan[key] for key in ("stage", "new_episodes", "reference_episodes", "preflight_episodes")}))


if __name__ == "__main__":
    main()
