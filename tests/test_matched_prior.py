import json
from pathlib import Path

import numpy as np
import pytest

from dt_corl_delay.delay_belief import CensoredAckDelayEstimator
from dt_corl_delay.matched_prior import matched_delay_prior, matched_delay_priors
from dt_corl_delay.prepare_cadr_ablation_config import materialize


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = {"delays": [0, 3], "probabilities": [0.3, 0.7]}
LATE = {"delays": [1, 4], "probabilities": [0.5, 0.5]}


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ({"type": "fixed", "delay": 3}, [0, 0, 0, 1, 0]),
        ({"type": "categorical", **REFERENCE}, [0.3, 0, 0, 0.7, 0]),
        (
            {"type": "changepoint", "before": REFERENCE, "after": LATE, "change_step": 3},
            [0.3, 0, 0, 0.7, 0],
        ),
        (
            {
                "type": "drifting",
                "delays": [0, 3, 4],
                "start_probabilities": [0.3, 0.7, 0.0],
                "end_probabilities": [0.0, 0.5, 0.5],
                "horizon_steps": 8,
            },
            [0.3, 0, 0, 0.7, 0],
        ),
        (
            {
                "type": "markov",
                "regimes": [REFERENCE, LATE],
                "transition": [[0.0, 1.0], [0.0, 1.0]],
            },
            [0.3, 0, 0, 0.7, 0],
        ),
    ],
)
def test_each_synthetic_scenario_uses_the_predeclared_reference(scenario, expected):
    assert matched_delay_prior(scenario, 4) == pytest.approx(expected, abs=1e-12)


def test_rejects_trace_without_using_its_samples():
    with pytest.raises(ValueError, match="no predeclared reference"):
        matched_delay_prior({"type": "trace", "path": "never-opened.csv"}, 4)


def test_rejects_reference_support_outside_estimator_bound():
    with pytest.raises(ValueError, match="outside 0..2"):
        matched_delay_prior({"type": "categorical", **REFERENCE}, 2)


def test_exact_sparse_fixed_prior_does_not_update():
    prior = matched_delay_prior({"type": "fixed", "delay": 2}, 4)
    estimator = CensoredAckDelayEstimator(max_delay=4, fixed_prior_probabilities=prior)
    assert np.array_equal(estimator.probabilities, prior)
    assert np.array_equal(estimator.effective_age_probabilities(5), prior)
    with pytest.raises(ValueError, match="cannot update"):
        estimator.update(5, [])
    estimator.reset()
    assert np.array_equal(estimator.probabilities, prior)
    with pytest.raises(ValueError, match="finite and positive"):
        CensoredAckDelayEstimator(max_delay=4, prior_probabilities=prior)


@pytest.mark.parametrize(
    ("config_name", "method"),
    [
        ("generated_formal_v1_hopper_cadr_seed101.json", "cadr_iql_ablation_fixed_prior_matched"),
        (
            "generated_formal_v1_walker2d_cadr_seed101.json",
            "cadr_iql_ablation_fixed_prior_matched",
        ),
        (
            "generated_formal_v1_pickcube_synthetic_seed101.json",
            "cadr_dp_ablation_fixed_prior_matched_plus_official_maniskill_dp",
        ),
    ],
)
def test_materialized_variants_keep_frozen_scenarios(tmp_path, config_name, method):
    template = ROOT / "configs" / config_name
    source = json.loads(template.read_text(encoding="utf-8"))
    matched = materialize(template, tmp_path / "matched.json", "fixed_prior_matched")
    legacy = materialize(template, tmp_path / "legacy.json", "fixed_prior")

    assert matched["experiment_series"].startswith("ablation_v2_")
    assert legacy["experiment_series"].startswith("ablation_v1_")
    assert matched["scenarios"] == legacy["scenarios"] == source["scenarios"]
    assert not matched["posterior_updates"]
    assert not legacy["posterior_updates"]
    assert method in matched["required_methods"]
    assert matched_delay_priors(legacy) == {}
    priors = matched_delay_priors(matched)
    assert priors.keys() == {scenario["name"] for scenario in source["scenarios"]}
    assert np.array_equal(priors["clean_delay_0"], [1.0] + [0.0] * source["max_delay"])
    assert np.array_equal(
        priors["fixed_delay_4"][:5], [0.0, 0.0, 0.0, 0.0, 1.0]
    )
    with pytest.raises(ValueError, match="requires posterior_updates=false"):
        matched_delay_priors({**matched, "posterior_updates": True})
