"""Shared opt-in method selection for checkpoint evaluation."""

from __future__ import annotations

from typing import Any, Mapping


def select_evaluation_methods(
    config: Mapping[str, Any],
    available: tuple[str, ...],
    aliases: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Resolve config selectors in the evaluator's original method order."""
    if "evaluation_methods" not in config:
        return available
    requested = config["evaluation_methods"]
    if (
        not isinstance(requested, list)
        or not requested
        or any(not isinstance(name, str) or not name for name in requested)
    ):
        raise ValueError("evaluation_methods must be a nonempty list of method keys")
    aliases = aliases or {}
    resolved = [aliases.get(name, name) for name in requested]
    for name, key in zip(requested, resolved):
        if key not in available:
            raise ValueError(
                f"evaluation method {name!r} is unavailable; choose from {', '.join(available)}"
            )
    if len(set(resolved)) != len(resolved):
        raise ValueError("evaluation_methods contains duplicate method keys")
    return tuple(key for key in available if key in resolved)
