from hashlib import sha256
import importlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def generator(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/revision"))
    return importlib.import_module("prepare_p2b1")


def test_reviewed_offsets_reuse_the_exact_p2a_pairing(generator, tmp_path):
    plan = generator.prepare(ROOT, tmp_path)
    assert plan == json.loads((tmp_path / "manifest.json").read_bytes())
    assert plan["stage"] == "p2b1" and plan["schema_version"] == 1
    assert [plan[field] for field in ("new_episodes", "reference_episodes", "preflight_episodes")] == [800, 200, 20]
    assert plan["decision_scenarios"] == [
        "fixed_delay_4", "fixed_delay_8", "markov_switching", "ood_support_shift",
    ]
    assert plan["descriptive_scenarios"] == ["iid_in_support"]
    assert [(run["seed"], run["offset"]) for run in plan["runs"]] == [
        (seed, offset) for seed in (101, 102) for offset in (-2, -1, 1, 2)
    ]
    preflight = plan["preflight"]
    assert (preflight["run_id"], preflight["seed"], preflight["offset"], preflight["scenarios"]) == (
        "hopper_seed101_offset_0", 101, 0, ["fixed_delay_8"],
    )

    p2a_path = ROOT / "results/revision/T6/p2a_plan.json"
    canonical_bytes = p2a_path.read_bytes().replace(b"\r\n", b"\n")
    assert sha256(canonical_bytes).hexdigest() == plan["p2a_plan_sha256"]
    p2a = json.loads(canonical_bytes)
    hopper = {batch["training_seed"]: batch for batch in p2a["batches"] if batch["task"] == "hopper"}
    for run in [*plan["runs"], preflight]:
        batch = hopper[run["seed"]]
        archived = generator.source_config(ROOT, batch)
        raw = (tmp_path / run["config"]).read_bytes()
        assert sha256(raw).hexdigest() == run["config_sha256"]
        assert run["inputs"] == batch["inputs"]
        names = (["fixed_delay_8"] if run is preflight else [
            "fixed_delay_4", "fixed_delay_8", "iid_in_support", "markov_switching", "ood_support_shift",
        ])
        assert run["scenarios"] == names
        assert json.loads(raw) == dict(
            archived,
            scenarios=[scene for scene in archived["scenarios"] if scene["name"] in names],
            experiment_series="ablation_v2_p2b1_hopper",
            evaluation_methods=["cadr_oracle_age_offset"],
            required_methods=["cadr_oracle_age_offset"],
            include_cadr_oracle_age=False,
            oracle_age_offset=run["offset"],
        )
        assert archived["eval_seed"] == 31000001 and archived["eval_episodes"] == 20
    assert generator.prepare(ROOT, tmp_path) == plan


def test_changed_published_p2a_plan_is_rejected(generator, tmp_path):
    source = tmp_path / generator.P2A_PLAN
    source.parent.mkdir(parents=True)
    source.write_text('{"stage":"p2a","batches":[]}\n', encoding="utf-8")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="published P2a input digest differs"):
        generator.prepare(tmp_path, output)
    assert not output.exists()
