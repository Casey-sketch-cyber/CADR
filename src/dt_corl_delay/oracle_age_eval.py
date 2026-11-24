"""Evaluation-only, privileged comparator using the selected packet's age."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .channel import PolicyInput
from .controllers import CadrController


CADR_ORACLE_AGE_METHOD = "cadr_oracle_age"


def include_cadr_oracle_age(config: Mapping[str, Any]) -> bool:
    """Keep the new comparator out of the archived v1 series."""

    enabled = bool(config.get("include_cadr_oracle_age", False))
    if enabled and not str(config.get("experiment_series", "")).startswith(
        ("formal_v2_", "ablation_v2_")
    ):
        raise ValueError("the privileged CADR oracle requires a separate v2 series")
    return enabled


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

    def _age_decision(
        self,
        policy_input: PolicyInput[np.ndarray],
        decision_step: int,
        raw_observation: np.ndarray,
        delay_probabilities: np.ndarray,
    ) -> tuple[np.ndarray, int, float, bool]:
        age = self._oracle_age
        self._oracle_age = None
        if age is None:
            raise RuntimeError("the evaluator must supply oracle age for every action")
        if age > decision_step or age > len(self._actions):
            raise ValueError("oracle age exceeds the executed action history")
        probabilities = np.zeros(age + 1, dtype=np.float64)
        probabilities[age] = 1.0
        return probabilities, age, 0.0, True
