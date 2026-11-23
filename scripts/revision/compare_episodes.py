"""Check reproduction of frozen-policy evaluations against archived episodes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True, order=True)
class EpisodeKey:
    task: str
    training_seed: int
    scenario: str
    method_key: str
    episode_index: int
    episode_seed: int


@dataclass(frozen=True)
class Outcome:
    episode_return: float
    success: int | None


def load_episodes(
    path: Path,
    *,
    task: str,
    training_seed: int,
    scenarios: set[str],
    methods: set[str],
) -> dict[EpisodeKey, Outcome]:
    """Read matching rows and reject ambiguous or malformed pairings."""
    required = set(EpisodeKey.__dataclass_fields__) | {"return", "success"}
    episodes: dict[EpisodeKey, Outcome] = {}
    indices: set[tuple[str, str, int]] = set()
    seeds: set[tuple[str, str, int]] = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing = required - set(fields)
        if missing or len(set(fields)) != len(fields):
            raise ValueError(f"{path}: invalid CSV header, missing columns {sorted(missing)}")
        for row in reader:
            if None in row or any(row[name] is None for name in required):
                raise ValueError(f"{path}:{reader.line_num}: malformed CSV row")
            if row["task"] != task:
                continue
            try:
                row_seed = int(row["training_seed"])
                scenario = row["scenario"]
                method = row["method_key"]
                if row_seed != training_seed:
                    continue
                if (scenarios and scenario not in scenarios) or (methods and method not in methods):
                    continue
                if not scenario or not method:
                    raise ValueError("scenario and method_key must be nonempty")
                index = int(row["episode_index"])
                seed = int(row["episode_seed"])
                if index < 0 or seed < 0:
                    raise ValueError("episode_index and episode_seed must be nonnegative")
                episode_return = float(row["return"])
                if not math.isfinite(episode_return):
                    raise ValueError("return must be finite")
                success = float(row["success"]) if row["success"].strip() else None
                if success is not None and success not in (0, 1):
                    raise ValueError("success must be blank, 0, or 1")
                key = EpisodeKey(task, row_seed, scenario, method, index, seed)
                index_key = (scenario, method, index)
                seed_key = (scenario, method, seed)
                if index_key in indices or seed_key in seeds:
                    raise ValueError(
                        f"duplicate episode index or seed for {scenario}/{method}: "
                        f"index={index}, seed={seed}"
                    )
            except ValueError as error:
                raise ValueError(f"{path}:{reader.line_num}: {error}") from error
            indices.add(index_key)
            seeds.add(seed_key)
            episodes[key] = Outcome(episode_return, None if success is None else int(success))
    return episodes


def compare_episodes(
    reference: dict[EpisodeKey, Outcome],
    candidate: dict[EpisodeKey, Outcome],
    *,
    abs_tol: float = 0.0,
    rel_tol: float = 0.0,
) -> tuple[dict, list[dict]]:
    """Return summary statistics and a record for every paired episode."""
    if not all(math.isfinite(tol) and tol >= 0 for tol in (abs_tol, rel_tol)):
        raise ValueError("return tolerances must be finite and nonnegative")
    pairs = []
    for key in sorted(reference.keys() & candidate.keys()):
        expected = reference[key]
        actual = candidate[key]
        pairs.append(
            {
                **asdict(key),
                "reference_return": expected.episode_return,
                "candidate_return": actual.episode_return,
                "return_delta": actual.episode_return - expected.episode_return,
                "return_matches": math.isclose(
                    actual.episode_return, expected.episode_return,
                    abs_tol=abs_tol, rel_tol=rel_tol,
                ),
                "reference_success": expected.success,
                "candidate_success": actual.success,
                "success_matches": expected.success == actual.success,
            }
        )
    missing = sorted(reference.keys() - candidate.keys())
    extra = sorted(candidate.keys() - reference.keys())
    deltas = [row["return_delta"] for row in pairs]
    return_mismatches = sum(not row["return_matches"] for row in pairs)
    success_mismatches = sum(not row["success_matches"] for row in pairs)
    summary = {
        "passed": bool(pairs) and not (missing or extra or return_mismatches or success_mismatches),
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "paired_count": len(pairs),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "missing_from_candidate": [asdict(key) for key in missing],
        "extra_in_candidate": [asdict(key) for key in extra],
        "return_abs_tol": abs_tol,
        "return_rel_tol": rel_tol,
        "return_nonzero_deltas": sum(delta != 0 for delta in deltas),
        "return_mismatches": return_mismatches,
        "return_max_abs_delta": max(map(abs, deltas), default=None),
        "return_mean_delta": math.fsum(deltas) / len(deltas) if deltas else None,
        "success_pairs_with_values": sum(
            row["reference_success"] is not None and row["candidate_success"] is not None
            for row in pairs
        ),
        "success_mismatches": success_mismatches,
    }
    return summary, pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="archived episodes.csv")
    parser.add_argument("candidate", type=Path, help="rerun episodes.csv")
    parser.add_argument("--task", required=True, help="exact task identifier")
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--scenario", action="append", help="repeat to compare selected scenarios")
    parser.add_argument("--method", action="append", help="repeat to compare selected method keys")
    parser.add_argument("--abs-tol", type=float, default=0.0, help="absolute return tolerance")
    parser.add_argument("--rel-tol", type=float, default=0.0, help="relative return tolerance")
    parser.add_argument("--output", type=Path, help="save the full report to a new JSON file")
    args = parser.parse_args(argv)
    selectors = {
        "task": args.task,
        "training_seed": args.training_seed,
        "scenarios": set(args.scenario or []),
        "methods": set(args.method or []),
    }
    try:
        reference = load_episodes(args.reference, **selectors)
        candidate = load_episodes(args.candidate, **selectors)
        missing_scenarios = selectors["scenarios"] - {key.scenario for key in reference}
        missing_methods = selectors["methods"] - {key.method_key for key in reference}
        if missing_scenarios or missing_methods:
            raise ValueError(
                f"reference selection missing scenarios {sorted(missing_scenarios)} "
                f"or methods {sorted(missing_methods)}"
            )
        summary, pairs = compare_episodes(
            reference, candidate, abs_tol=args.abs_tol, rel_tol=args.rel_tol
        )
        summary["selection"] = {
            "task": args.task,
            "training_seed": args.training_seed,
            "scenarios": sorted(selectors["scenarios"]),
            "methods": sorted(selectors["methods"]),
        }
        report = {"summary": summary, "episodes": pairs}
        if args.output:
            with args.output.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(report, handle, indent=2, allow_nan=False)
                handle.write("\n")
        print(json.dumps(summary if args.output else report, indent=2, allow_nan=False))
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
