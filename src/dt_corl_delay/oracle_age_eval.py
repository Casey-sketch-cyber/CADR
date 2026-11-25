"""Evaluation-only, privileged comparator using the selected packet's age."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .channel import PolicyInput
from .controllers import CadrController


CADR_ORACLE_AGE_METHOD = "cadr_oracle_age"
CADR_ORACLE_AGE_OFFSET_METHOD = "cadr_oracle_age_offset"
P2B1_SERIES_PREFIX = "ablation_v2_p2b1_"


def include_cadr_oracle_age(config: Mapping[str, Any]) -> bool:
    """Keep the new comparator out of the archived v1 series."""

    enabled = bool(config.get("include_cadr_oracle_age", False))
    if enabled and not str(config.get("experiment_series", "")).startswith(
        ("formal_v2_", "ablation_v2_")
    ):
        raise ValueError("the privileged CADR oracle requires a separate v2 series")
    return enabled


def _validate_oracle_age_offset(offset: object) -> int:
    if type(offset) is not int or offset not in (-2, -1, 0, 1, 2):
        raise ValueError("oracle_age_offset must be an integer in {-2, -1, 0, 1, 2}")
    return offset


def configured_oracle_age_offset(config: Mapping[str, Any]) -> int | None:
    """Validate the separate, evaluator-only P2b-1 method request."""
    if "oracle_age_offset" not in config:
        return None
    offset = _validate_oracle_age_offset(config["oracle_age_offset"])
    series = config.get("experiment_series")
    if not isinstance(series, str) or not series.startswith(P2B1_SERIES_PREFIX) or series == P2B1_SERIES_PREFIX:
        raise ValueError(f"oracle_age_offset requires a dedicated {P2B1_SERIES_PREFIX}* series")
    if config.get("evaluation_methods") != [CADR_ORACLE_AGE_OFFSET_METHOD]:
        raise ValueError("oracle_age_offset requires evaluation_methods=['cadr_oracle_age_offset']")
    if config.get("include_cadr_oracle_age") or config.get("include_oracle_upper_bound"):
        raise ValueError("oracle_age_offset uses a separate config from other oracle methods")
    return offset


class OracleAgeCadrController(CadrController):
    """Privileged CADR comparator. Only the evaluator supplies the true age."""

    _oracle_age: int | None = None

    def reset(self, initial_observation: np.ndarray) -> None:
        super().reset(initial_observation)
        self._oracle_age = None

    def set_oracle_age(self, age: int) -> None:
        if age < 0:
            raise ValueError("oracle age cannot be negative")
        self._oracle_age = int(age)

    def _selected_age_for_decision(self, decision_step: int) -> int:
        age = self._oracle_age
        self._oracle_age = None
        if age is None:
            raise RuntimeError("the evaluator must supply oracle age for every action")
        if age > decision_step or age > len(self._actions):
            raise ValueError("oracle age exceeds the executed action history")
        return age

    def _age_decision(
        self,
        policy_input: PolicyInput[np.ndarray],
        decision_step: int,
        raw_observation: np.ndarray,
        delay_probabilities: np.ndarray,
    ) -> tuple[np.ndarray, int, float, bool]:
        age = self._selected_age_for_decision(decision_step)
        probabilities = np.zeros(age + 1, dtype=np.float64)
        probabilities[age] = 1.0
        return probabilities, age, 0.0, True


class OracleAgeOffsetCadrController(OracleAgeCadrController):
    """Use a deliberately perturbed age for the privileged P2b-1 diagnostic."""

    def __init__(self, *, oracle_age_offset: int, **controller_kwargs: Any) -> None:
        self.oracle_age_offset = _validate_oracle_age_offset(oracle_age_offset)
        super().__init__(**controller_kwargs)

    def reset(self, initial_observation: np.ndarray) -> None:
        super().reset(initial_observation)
        self._offset_decisions = 0
        self._effective_offset_sum = 0
        self._fully_clipped_steps = 0

    def _selected_age_for_decision(self, decision_step: int) -> int:
        true_age = super()._selected_age_for_decision(decision_step)
        selected_age = min(min(decision_step, self.max_delay), max(0, true_age + self.oracle_age_offset))
        if selected_age > len(self._actions):
            raise ValueError("offset oracle age exceeds the executed action history")
        effective_offset = selected_age - true_age
        self._offset_decisions += 1
        self._effective_offset_sum += effective_offset
        self._fully_clipped_steps += int(self.oracle_age_offset != 0 and effective_offset == 0)
        return selected_age

    def age_offset_counts(self) -> tuple[int, int, int]:
        """Return per-episode decision, effective offset, and full clipping totals."""
        return self._offset_decisions, self._effective_offset_sum, self._fully_clipped_steps
