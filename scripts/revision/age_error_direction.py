"""Policy-free diagnostic of CADR age inference vs the in-order rule.

Replicates the controller's age inference with the v1 estimator settings and a
per-episode reset, and scores each changed-observation step against the true
observation age, split by time within the episode.
"""
import json
import random
import sys

import numpy as np

sys.path.insert(0, sys.argv[1])
from dt_corl_delay.channel import DelayedObservationChannel  # noqa: E402
from dt_corl_delay.delay_belief import (  # noqa: E402
    CensoredAckDelayEstimator,
    ContentConditionedAgePosterior,
)
from dt_corl_delay.delay_process import build_delay_process  # noqa: E402

CONFIG = json.load(open(sys.argv[2]))
EPISODES = int(sys.argv[3])
HORIZON = 1000
BUCKETS = [(1, 50), (50, 200), (200, 1000)]
METHODS = ("hazard", "legacy_cdf")


def run(scenario):
    stats = {}
    for episode in range(EPISODES):
        seed = 1000 + 100_003 * episode
        channel = DelayedObservationChannel(
            build_delay_process(scenario), random.Random(seed + 1), ack_lag=1
        )
        channel.reset(np.array([0.0]))
        estimators, posteriors = {}, {}
        for name in METHODS:
            estimators[name] = CensoredAckDelayEstimator(
                max_delay=32, ack_lag=1, memory_steps=128, decay_time=48.0, prior_strength=2.0
            )
            estimators[name].reset()
            posteriors[name] = ContentConditionedAgePosterior(32, name)
            posteriors[name].reset(np.array([0.0]))
        latest_revealed = 0
        previous = np.array([0.0])
        for t in range(HORIZON):
            step = channel.advance(t)
            acks = step.policy_input.acknowledgements
            selected_generation = step.selected_packet.generated_step
            true_age = t - selected_generation
            observation = np.array([float(selected_generation)])
            for ack in acks:
                latest_revealed = max(latest_revealed, int(ack.generated_step))
            changed = not np.array_equal(observation, previous)
            in_order = min(32, max(0, t - (latest_revealed + int(changed)))) if t > 0 else 0
            beliefs = {}
            for name in METHODS:
                q = estimators[name].update(t, acks)
                base = estimators[name].effective_age_probabilities(t)
                beliefs[name], _, _ = posteriors[name].probabilities(observation, t, acks, base, q)
            previous = observation
            if t > 0 and changed:
                bucket = next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= t < hi)
                s = stats.setdefault(bucket, {"n": 0})
                s["n"] += 1
                s["in_order_acc"] = s.get("in_order_acc", 0) + (in_order == true_age)
                s["in_order_err"] = s.get("in_order_err", 0) + abs(in_order - true_age)
                for name, p in beliefs.items():
                    ages = np.arange(len(p))
                    s[name + "_map_acc"] = s.get(name + "_map_acc", 0) + (int(np.argmax(p)) == true_age)
                    s[name + "_mean_err"] = s.get(name + "_mean_err", 0) + abs(float(ages @ p) - true_age)
                    s[name + "_bias"] = s.get(name + "_bias", 0) + (float(ages @ p) - true_age)
            channel.send(np.array([float(t + 1)]), generated_step=t + 1)
    return stats


names = sys.argv[4].split(",")
print(f"{'scenario':24s} {'steps':>9s} {'n':>6s} | {'inord acc':>9s} {'err':>5s} | "
      f"{'haz MAP':>7s} {'|mean-true|':>11s} {'bias':>6s} | {'leg MAP':>7s} {'|mean-true|':>11s}")
for scenario in CONFIG["scenarios"]:
    if scenario["name"] not in names:
        continue
    stats = run(scenario)
    for bucket, s in sorted(stats.items()):
        n = s["n"]
        lo, hi = BUCKETS[bucket]
        print(f"{scenario['name']:24s} {lo:4d}-{hi:<4d} {n:6d} | "
              f"{s['in_order_acc'] / n:9.2f} {s['in_order_err'] / n:5.2f} | "
              f"{s['hazard_map_acc'] / n:7.2f} {s['hazard_mean_err'] / n:11.2f} {s['hazard_bias'] / n:6.2f} | "
              f"{s['legacy_cdf_map_acc'] / n:7.2f} {s['legacy_cdf_mean_err'] / n:11.2f}")
