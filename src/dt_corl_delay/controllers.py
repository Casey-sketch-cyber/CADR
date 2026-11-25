"""Deployment controllers that obey the latent-delay information boundary."""

from __future__ import annotations

import math
import statistics
from numbers import Integral
from typing import Any, Sequence

import numpy as np
import torch

from .channel import PolicyInput
from .delay_belief import (
    CensoredAckDelayEstimator,
    ContentConditionedAgePosterior,
    validate_age_posterior,
)
from .dynamics import DeltaDynamicsEnsemble
from .simple_baselines import SIMPLE_AGE_DECISION_RULES
from .upper_quantile import upper_age_quantile, validate_age_quantile


def state_disagreement_components(
    age_states: torch.Tensor,
    age_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose posterior state spread into epistemic and age terms.

    ``age_states`` has shape ``[age, ensemble, state]`` and ``age_weights``
    is a normalized posterior over age. The returned root-mean-square terms
    obey the law of total variance (up to floating-point round-off): total
    variance is epistemic variance plus between-age variance.
    """

    if age_states.ndim != 3:
        raise ValueError("age_states must have shape [age, ensemble, state]")
    if age_weights.ndim != 1 or age_weights.shape[0] != age_states.shape[0]:
        raise ValueError("age_weights must match the age dimension")
    if not torch.isclose(age_weights.sum(), age_weights.new_tensor(1.0)):
        raise ValueError("age_weights must sum to one")
    age_means = age_states.mean(dim=1)
    posterior_mean = (age_means * age_weights[:, None]).sum(dim=0)
    epistemic_variance = (
        (age_states - age_means[:, None, :]).pow(2).mean(dim=-1).mean(dim=1)
        * age_weights
    ).sum()
    age_variance = (
        (age_means - posterior_mean).pow(2).mean(dim=-1) * age_weights
    ).sum()
    total_variance = (epistemic_variance + age_variance).clamp_min(0.0)
    return (
        torch.sqrt(total_variance),
        torch.sqrt(epistemic_variance.clamp_min(0.0)),
        torch.sqrt(age_variance.clamp_min(0.0)),
    )


def reconstruction_reliability(
    state_disagreement: float,
    risk_blend: float,
    risk_threshold: float,
) -> float:
    """Return the smooth dead-zone reliability used by robust blending."""

    excess_disagreement = max(0.0, state_disagreement - risk_threshold)
    return math.exp(-risk_blend * excess_disagreement)


class DirectObservationController:
    """Frozen IQL baseline that treats the received observation as current."""

    def __init__(
        self,
        agent: Any,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
    ) -> None:
        self.agent = agent
        self.state_mean = state_mean
        self.state_std = state_std

    def reset(self, initial_observation: np.ndarray) -> None:
        del initial_observation

    def act(self, policy_input: PolicyInput[np.ndarray], decision_step: int) -> np.ndarray:
        del decision_step
        return self.agent.act(
            np.asarray(policy_input.observation),
            self.state_mean,
            self.state_std,
        )

    def observe_transition(
        self,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        del action, reward, terminated, truncated

    def diagnostics(self) -> dict[str, float]:
        return {}


class CadrController:
    """Censored-ACK Delay Reconstruction (CADR) policy wrapper.

    By default the controller marginalizes frozen-actor actions over candidate
    current states reconstructed for every plausible age and dynamics member.
    Current packet delay, generation time, and channel internals are never
    passed to this class.
    """

    def __init__(
        self,
        agent: Any,
        dynamics: DeltaDynamicsEnsemble,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        max_delay: int,
        ack_lag: int,
        memory_steps: int = 128,
        decay_time: float = 48.0,
        prior_strength: float = 2.0,
        risk_blend: float = 0.0,
        risk_threshold: float = 0.0,
        age_decision_rule: str = "posterior_mean",
        entropy_gate_threshold: float = 1.4,
        use_content_conditioning: bool = True,
        posterior_updates: bool = True,
        use_censoring: bool = True,
        age_posterior: str = "legacy_cdf",
        fixed_prior_probabilities: Sequence[float] | None = None,
        fixed_horizon: int | None = None,
        aggregation: str = "action_mean",
        ensemble_members: Sequence[int] | None = None,
        age_quantile: float | None = None,
    ) -> None:
        if not math.isfinite(risk_blend) or risk_blend < 0.0:
            raise ValueError("risk_blend must be finite and non-negative")
        if not math.isfinite(risk_threshold) or risk_threshold < 0.0:
            raise ValueError("risk_threshold must be finite and non-negative")
        supported_rules = {"posterior_mean", "map", "entropy_gated", "upper_quantile"} | SIMPLE_AGE_DECISION_RULES
        if age_decision_rule not in supported_rules:
            raise ValueError("unsupported age_decision_rule")
        if aggregation not in {"action_mean", "state_mean", "map"}:
            raise ValueError("unsupported aggregation")
        if age_decision_rule == "upper_quantile":
            age_quantile = validate_age_quantile(age_quantile)
            if aggregation != "action_mean":
                raise ValueError("upper_quantile requires action_mean aggregation")
            if age_posterior != "hazard" or not use_content_conditioning:
                raise ValueError("upper_quantile requires the content-conditioned hazard posterior")
            if fixed_prior_probabilities is not None:
                raise ValueError("upper_quantile requires the positive support of the standard delay prior")
        if aggregation == "map":
            if age_decision_rule not in {"posterior_mean", "map"}:
                raise ValueError("map aggregation conflicts with age_decision_rule")
            age_decision_rule = "map"
        if ensemble_members is not None:
            if not isinstance(ensemble_members, Sequence) or not ensemble_members or any(
                isinstance(index, bool)
                or not isinstance(index, Integral)
                or not 0 <= index < dynamics.ensemble_size
                for index in ensemble_members
            ):
                raise ValueError("ensemble_members must contain valid integer indices")
            if len(set(ensemble_members)) != len(ensemble_members):
                raise ValueError("ensemble_members must not contain duplicates")
        if entropy_gate_threshold < 0.0:
            raise ValueError("entropy_gate_threshold must be non-negative")
        validate_age_posterior(age_posterior, ack_lag, use_content_conditioning)
        if max_delay < 0:
            raise ValueError("max_delay must be non-negative")
        if age_decision_rule == "in_order_age" and (
            ack_lag != 1 or not use_content_conditioning
        ):
            raise ValueError("in_order_age requires ack_lag=1 and content conditioning")
        if age_decision_rule == "fixed_horizon_model":
            if type(fixed_horizon) is not int or not 0 <= fixed_horizon <= max_delay:
                raise ValueError(
                    "fixed_horizon_model requires an integer horizon in 0..max_delay"
                )
        elif fixed_horizon is not None:
            raise ValueError("fixed_horizon is only used by fixed_horizon_model")
        self.agent = agent
        self.dynamics = dynamics
        self.state_mean = state_mean
        self.state_std = state_std
        self.risk_blend = risk_blend
        self.risk_threshold = risk_threshold
        self.aggregation = aggregation
        self.ensemble_members = (
            tuple(int(index) for index in ensemble_members)
            if ensemble_members is not None
            else None
        )
        self.age_decision_rule = age_decision_rule
        self.age_quantile = age_quantile
        self.entropy_gate_threshold = entropy_gate_threshold
        self.use_content_conditioning = bool(use_content_conditioning)
        self.posterior_updates = bool(posterior_updates)
        if fixed_prior_probabilities is not None and self.posterior_updates:
            raise ValueError("a fixed delay prior requires posterior_updates=false")
        if fixed_prior_probabilities is not None and age_decision_rule in SIMPLE_AGE_DECISION_RULES:
            raise ValueError("simple age rules do not use a matched delay prior")
        self.max_delay = max_delay
        self.fixed_horizon = fixed_horizon
        self.estimator = None
        if age_decision_rule not in SIMPLE_AGE_DECISION_RULES:
            self.estimator = CensoredAckDelayEstimator(
                max_delay=max_delay,
                ack_lag=ack_lag,
                memory_steps=memory_steps,
                decay_time=decay_time,
                prior_strength=prior_strength,
                use_censoring=use_censoring,
                fixed_prior_probabilities=fixed_prior_probabilities,
            )
        self._actions: list[torch.Tensor] = []
        self._diagnostics: dict[str, list[float]] = {}
        self._content_age_posterior = ContentConditionedAgePosterior(max_delay, age_posterior)
        self._known_selected_generated_step = 0
        self._previous_observation: np.ndarray | None = None

    def reset(self, initial_observation: np.ndarray) -> None:
        if self.estimator is not None:
            self.estimator.reset()
        self._actions.clear()
        if self.estimator is not None:
            self._content_age_posterior.reset(initial_observation)
        if self.age_decision_rule == "in_order_age":
            self._known_selected_generated_step = 0
            self._previous_observation = np.asarray(initial_observation).copy()
        self._diagnostics = {
            "posterior_delay_mean": [],
            "effective_age_mean": [],
            "posterior_entropy": [],
            "state_disagreement": [],
            "epistemic_state_disagreement": [],
            "age_hypothesis_state_disagreement": [],
            "action_disagreement": [],
            "exact_age_from_unchanged_observation": [],
            "observation_changed": [],
            "selected_age": [],
            "conditioned_age_entropy": [],
            "map_rule_used": [],
            "reconstruction_reliability": [],
        }
        if self.aggregation == "state_mean":
            del self._diagnostics["action_disagreement"]
        if self.estimator is None:
            del self._diagnostics["posterior_delay_mean"]
            del self._diagnostics["posterior_entropy"]
        if self.fixed_horizon is not None:
            del self._diagnostics["exact_age_from_unchanged_observation"]
            del self._diagnostics["observation_changed"]

    def _content_conditioned_age_probabilities(
        self,
        observation: np.ndarray,
        decision_step: int,
        base_probabilities: np.ndarray,
        policy_input: PolicyInput[np.ndarray],
        delay_probabilities: np.ndarray,
    ) -> np.ndarray:
        probabilities, exact_age, changed = self._content_age_posterior.probabilities(
            observation,
            decision_step,
            policy_input.acknowledgements,
            base_probabilities,
            delay_probabilities,
        )
        self._diagnostics["exact_age_from_unchanged_observation"].append(float(exact_age))
        self._diagnostics["observation_changed"].append(float(changed))
        if self._content_age_posterior.age_posterior == "hazard":
            self._diagnostics.setdefault("hazard_zero_mass_fallback", []).append(
                float(self._content_age_posterior.hazard_zero_mass_fallback)
            )
        return probabilities

    def _in_order_age(
        self,
        policy_input: PolicyInput[np.ndarray],
        decision_step: int,
        observation: np.ndarray,
    ) -> int:
        for acknowledgement in policy_input.acknowledgements:
            if acknowledgement.revealed_step > decision_step:
                raise ValueError("an ACK cannot be used before its reveal step")
            self._known_selected_generated_step = max(
                self._known_selected_generated_step,
                int(acknowledgement.generated_step),
            )
        changed = (
            self._previous_observation is not None
            and not np.array_equal(observation, self._previous_observation)
        )
        generated_step = self._known_selected_generated_step + int(changed)
        self._previous_observation = observation.copy()
        self._diagnostics["exact_age_from_unchanged_observation"].append(
            float(decision_step == 0 or not changed)
        )
        self._diagnostics["observation_changed"].append(float(changed))
        return min(self.max_delay, max(0, decision_step - generated_step))

    def _age_decision(
        self,
        policy_input: PolicyInput[np.ndarray],
        decision_step: int,
        raw_observation: np.ndarray,
        delay_probabilities: np.ndarray | None,
    ) -> tuple[np.ndarray, int, float, bool]:
        if self.estimator is None:
            if self.fixed_horizon is not None:
                age = min(self.fixed_horizon, decision_step, len(self._actions))
            else:
                age = self._in_order_age(policy_input, decision_step, raw_observation)
            probabilities = np.zeros(age + 1, dtype=np.float64)
            probabilities[age] = 1.0
            return probabilities, age, 0.0, False
        assert delay_probabilities is not None
        age_probabilities = self.estimator.effective_age_probabilities(decision_step)
        # Without content, the observation-change constraint on G_t is absent.
        if self.use_content_conditioning:
            age_probabilities = self._content_conditioned_age_probabilities(
                raw_observation,
                decision_step,
                age_probabilities,
                policy_input,
                delay_probabilities,
            )
        selected_age = int(np.argmax(age_probabilities))
        positive_probabilities = age_probabilities[age_probabilities > 0.0]
        conditioned_age_entropy = float(
            -(positive_probabilities * np.log(positive_probabilities)).sum()
        )
        if self.age_decision_rule == "upper_quantile":
            assert self.age_quantile is not None
            selected_age = upper_age_quantile(age_probabilities, self.age_quantile)
            point_mass = np.zeros(selected_age + 1, dtype=np.float64)
            point_mass[selected_age] = 1.0
            return point_mass, selected_age, conditioned_age_entropy, False
        use_map = self.age_decision_rule == "map" or (
            self.age_decision_rule == "entropy_gated"
            and self.estimator.entropy >= self.entropy_gate_threshold
        )
        if use_map:
            age_probabilities = np.zeros_like(age_probabilities)
            age_probabilities[selected_age] = 1.0
        return age_probabilities, selected_age, conditioned_age_entropy, use_map

    def _update_delay_statistics(
        self,
        policy_input: PolicyInput[np.ndarray],
        decision_step: int,
    ) -> np.ndarray | None:
        if self.estimator is None:
            return None
        if self.posterior_updates:
            return self.estimator.update(decision_step, policy_input.acknowledgements)
        return self.estimator.probabilities

    def _record_delay_statistics(self, delay_probabilities: np.ndarray | None) -> None:
        if delay_probabilities is not None:
            support = np.arange(delay_probabilities.size, dtype=np.float64)
            self._diagnostics["posterior_delay_mean"].append(
                float(delay_probabilities @ support)
            )
            assert self.estimator is not None
            self._diagnostics["posterior_entropy"].append(self.estimator.entropy)

    @torch.inference_mode()
    def act(self, policy_input: PolicyInput[np.ndarray], decision_step: int) -> np.ndarray:
        delay_probabilities = self._update_delay_statistics(policy_input, decision_step)
        raw_observation = np.asarray(policy_input.observation)
        age_probabilities, selected_age, conditioned_age_entropy, use_map = (
            self._age_decision(policy_input, decision_step, raw_observation, delay_probabilities)
        )
        observation = torch.as_tensor(
            raw_observation,
            dtype=torch.float32,
            device=self.agent.device,
        ).unsqueeze(0)
        normalized_observation = (observation - self.state_mean) / self.state_std

        if len(age_probabilities) == 1 and age_probabilities[0] == 1.0:
            # Preserve bit-level equivalence with the direct policy in the
            # proven zero-delay case; continuous-control trajectories can be
            # chaotic under even round-off-sized action differences.
            action = self.agent.act_normalized(normalized_observation)
            state_disagreement = torch.zeros((), device=self.agent.device)
            epistemic_state_disagreement = torch.zeros(
                (), device=self.agent.device
            )
            age_hypothesis_state_disagreement = torch.zeros(
                (), device=self.agent.device
            )
            action_disagreement = (
                None
                if self.aggregation == "state_mean"
                else torch.zeros((), device=self.agent.device)
            )
        else:
            maximum_age = len(age_probabilities) - 1
            if self.ensemble_members is None:
                age_states = self.dynamics.rollout_age_candidates(
                    normalized_observation, self._actions, maximum_age=maximum_age
                )
                ensemble_size = self.dynamics.ensemble_size
            else:
                age_states = self.dynamics.rollout_age_candidates(
                    normalized_observation,
                    self._actions,
                    maximum_age=maximum_age,
                    member_indices=self.ensemble_members,
                )
                ensemble_size = len(self.ensemble_members)
            states = age_states.reshape(-1, self.dynamics.state_dim)
            age_weights = torch.as_tensor(
                age_probabilities,
                dtype=torch.float32,
                device=self.agent.device,
            )
            weights = age_weights.repeat_interleave(ensemble_size)
            weights = weights / ensemble_size
            if self.aggregation == "state_mean":
                mean_state = (states * weights[:, None]).sum(dim=0, keepdim=True)
                action = self.agent.act_normalized(mean_state)
                candidate_actions = None
            else:
                candidate_actions = self.agent.act_normalized(states)
                action = (candidate_actions * weights[:, None]).sum(dim=0, keepdim=True)

            (
                state_disagreement,
                epistemic_state_disagreement,
                age_hypothesis_state_disagreement,
            ) = state_disagreement_components(
                age_states,
                age_weights,
            )
            if candidate_actions is None:
                action_disagreement = None
            else:
                action_mean = (candidate_actions * weights[:, None]).sum(dim=0)
                action_disagreement = torch.sqrt(
                    ((candidate_actions - action_mean).pow(2).mean(dim=-1) * weights).sum()
                )
        reliability = 1.0
        if self.risk_blend > 0.0:
            direct_action = self.agent.act_normalized(normalized_observation)
            reliability = reconstruction_reliability(
                float(state_disagreement),
                self.risk_blend,
                self.risk_threshold,
            )
            action = reliability * action + (1.0 - reliability) * direct_action
        action = action.clamp(-1.0, 1.0)
        self._actions.append(action.detach())

        ages = np.arange(age_probabilities.size, dtype=np.float64)
        self._record_delay_statistics(delay_probabilities)
        self._diagnostics["effective_age_mean"].append(float(age_probabilities @ ages))
        self._diagnostics["state_disagreement"].append(float(state_disagreement))
        self._diagnostics["epistemic_state_disagreement"].append(
            float(epistemic_state_disagreement)
        )
        self._diagnostics["age_hypothesis_state_disagreement"].append(
            float(age_hypothesis_state_disagreement)
        )
        if action_disagreement is not None:
            self._diagnostics["action_disagreement"].append(float(action_disagreement))
        self._diagnostics["selected_age"].append(float(selected_age))
        self._diagnostics["conditioned_age_entropy"].append(conditioned_age_entropy)
        self._diagnostics["map_rule_used"].append(float(use_map))
        self._diagnostics["reconstruction_reliability"].append(float(reliability))
        return action.squeeze(0).cpu().numpy()

    def observe_transition(
        self,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        del action, reward, terminated, truncated

    def diagnostics(self) -> dict[str, float]:
        return {
            f"mean_{name}": statistics.fmean(values) if values else 0.0
            for name, values in self._diagnostics.items()
        }
