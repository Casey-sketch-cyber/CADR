import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np
import torch

from dt_corl_delay.delayed_eval import (
    _environment_horizon,
    _episode_success_summary,
    evaluate_controller,
)


class EnvironmentHorizonTest(unittest.TestCase):
    def test_success_summary_separates_once_end_and_post_change(self) -> None:
        summary = _episode_success_summary(
            [(0, False), (1, True), (2, False), (3, True)],
            {"type": "changepoint", "change_step": 2},
        )
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertEqual(summary["once"], 1.0)
        self.assertEqual(summary["at_end"], 1.0)
        self.assertEqual(summary["occupancy"], 0.5)
        self.assertEqual(summary["post_change_occupancy"], 0.5)

    def test_explicit_horizon_survives_wrapper_spec_without_limit(self) -> None:
        environment = SimpleNamespace(
            spec=SimpleNamespace(max_episode_steps=None),
            unwrapped=SimpleNamespace(spec=SimpleNamespace(max_episode_steps=None)),
        )
        self.assertEqual(
            _environment_horizon(environment, {"max_episode_steps": 50}),
            50,
        )

    def test_unwrapped_spec_is_a_valid_fallback(self) -> None:
        environment = SimpleNamespace(
            spec=SimpleNamespace(max_episode_steps=None),
            unwrapped=SimpleNamespace(spec=SimpleNamespace(max_episode_steps=1000)),
        )
        self.assertEqual(_environment_horizon(environment, None), 1000)


class ControllerTimingTest(unittest.TestCase):
    def _evaluate(self, controller_timing_device: torch.device | None):
        events: list[str] = []
        observation = np.zeros(1)

        def act(*_args, **_kwargs):
            events.append("act")
            return observation.copy()

        def step(_action):
            events.append("env.step")
            return observation, 1.0, False, False, {}

        environment = SimpleNamespace(
            spec=SimpleNamespace(max_episode_steps=2),
            reset=Mock(return_value=(observation, {})),
            step=step,
            close=Mock(),
        )
        controller = SimpleNamespace(
            reset=Mock(),
            act=act,
            observe_transition=Mock(),
            diagnostics=Mock(return_value={}),
        )
        timestamps = iter((1_000_000, 2_000_000, 5_000_000, 8_000_000))

        def clock():
            events.append("clock")
            return next(timestamps)

        def synchronize(_device):
            events.append("synchronize")

        with (
            patch(
                "dt_corl_delay.delayed_eval._recover_environment",
                return_value=environment,
            ),
            patch("dt_corl_delay.delayed_eval.time.perf_counter_ns", side_effect=clock),
            patch(
                "dt_corl_delay.delayed_eval.torch.cuda.synchronize",
                side_effect=synchronize,
            ) as cuda_sync,
        ):
            summary = evaluate_controller(
                controller_factory=lambda: controller,
                dataset_id="synthetic",
                scenario={"name": "clean", "type": "fixed", "delay": 0},
                episodes=1,
                seed=101,
                ack_lag=1,
                cvar_alpha=0.2,
                method="test",
                controller_timing_device=controller_timing_device,
            )
        self.assertEqual(summary["returns"], [2.0])
        self.assertEqual(summary["inference_ms_p50"], 2.0)
        self.assertAlmostEqual(summary["inference_ms_p95"], 2.9)
        return summary, events, cuda_sync

    def test_cuda_timing_brackets_controller_and_excludes_environment(self) -> None:
        device = torch.device("cuda:1")
        summary, events, cuda_sync = self._evaluate(device)
        self.assertEqual(cuda_sync.call_args_list, [call(device)] * 4)
        self.assertEqual(
            events,
            ["synchronize", "clock", "act", "synchronize", "clock", "env.step"] * 2,
        )
        self.assertEqual(summary["controller_timing"], {
            "scope": "controller.act",
            "device": "cuda:1",
            "cuda_synchronized": True,
            "samples": 2,
        })

    def test_cpu_and_legacy_timing_do_not_sync(self) -> None:
        legacy, _, legacy_sync = self._evaluate(None)
        legacy_sync.assert_not_called()
        self.assertNotIn("controller_timing", legacy)
        cpu, events, cpu_sync = self._evaluate(torch.device("cpu"))
        cpu_sync.assert_not_called()
        self.assertEqual(events, ["clock", "act", "clock", "env.step"] * 2)
        self.assertEqual(cpu.pop("controller_timing"), {
            "scope": "controller.act",
            "device": "cpu",
            "cuda_synchronized": False,
            "samples": 2,
        })
        self.assertEqual(cpu, legacy)


if __name__ == "__main__":
    unittest.main()
