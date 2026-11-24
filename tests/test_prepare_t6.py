from hashlib import sha256
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_t6_test", ROOT / "scripts/revision/prepare_t6.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_prepare_exact_reviewed_p2a_matrix(tmp_path):
    result = MODULE.prepare(ROOT, tmp_path)
    assert result["method_runs"] == 58 and result["expected_episodes"] == 13200
    expected = {
        ("hopper", 101): (14, 12), ("hopper", 102): (14, 12),
        ("hopper", 103): (6, 12), ("hopper", 104): (6, 12), ("hopper", 105): (6, 12),
        ("pickcube", 101): (6, 9), ("pickcube", 102): (6, 9),
    }
    assert {(b["task"], b["training_seed"]): (len(b["runs"]), len(b["scenarios"]))
            for b in result["batches"]} == expected
    for batch in result["batches"]:
        metadata = next(item for item in batch["inputs"] if item["role"] == "dataset_metadata")
        assert len(metadata["sha256"]) == 64
        if batch["task"] == "hopper":
            assert metadata["path"].endswith("/medium-v0/data/metadata.json")
        original = (ROOT / batch["source_config"]).read_bytes()
        assert sha256(original).hexdigest() == batch["source_config_sha256"]
        template = json.loads(original)
        sets = {}
        for run in batch["runs"]:
            contents = (tmp_path / run["config"]).read_bytes()
            assert sha256(contents).hexdigest() == run["config_sha256"]
            config = json.loads(contents)
            sets[run["method_id"]] = config
            assert config["scenarios"] == template["scenarios"]
            assert config["eval_seed"] == template["eval_seed"] and config["eval_episodes"] == 20
            assert config["evaluation_methods"] == [run["evaluation_key"]]
            assert not config["include_oracle_upper_bound"]
        assert sets["full_legacy"].get("age_posterior", "legacy_cdf") == "legacy_cdf"
        assert sets["full_hazard"]["age_posterior"] == "hazard"
        assert sets["no_censoring"]["use_censoring"] is False
        assert sets["cadr_oracle_age"]["include_cadr_oracle_age"] is True
        assert sum(r["method_id"] == "direct" for r in batch["runs"]) == 1
        if len(sets) == 14:
            assert sets["no_content"]["age_posterior"] == "legacy_cdf"
            assert sets["ack_map_hazard"]["aggregation"] == "map"
            assert sets["single_model"]["ensemble_members"] == [0]
            assert sets["single_model"]["ensemble_size"] == template["ensemble_size"]
            assert sets["full_hazard_state_mean"]["aggregation"] == "state_mean"
    assert MODULE.prepare(ROOT, tmp_path) == result
