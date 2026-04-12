"""PoC experiment — Causal Intervention Analysis.

Demonstrates whether controller internal state is causally active (drives
behaviour) vs. epiphenomenal (decorative).  This is a key capability test
for next-generation AI models where:

1. **Alignment safety** — systems must provably use internal state for
   decisions, not just correlate with them.
2. **Interpretability** — causal influence maps reveal which hidden
   dimensions actually matter.
3. **Planning depth** — temporal causal depth measures how far ahead a
   controller "looks" via its state dynamics.

Protocol:
  a) Run a baseline episode with a memory-bearing controller.
  b) At mid-episode, surgically perturb the hidden state by a known δ.
  c) Continue and compare behavioural divergence.
  d) Separately, shuffle the hidden state at every step to test
     epiphenomenality.
  e) Measure temporal causal depth from baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..body_reflex import BodylessBodyLayer
from ..controllers import MemoryController, ReducedDescendingController
from ..envs import FlyAvatarEnv
from ..experiments.benchmark_harness import run_episode
from ..metrics import summarize_metrics
from ..metrics.causal_metrics import (
    causal_influence_score,
    epiphenomenal_test,
    temporal_causal_depth,
)


_EPISODE_STEPS = 30
_SEEDS = [0, 1, 2]


def _run_baseline(seed: int) -> list:
    env = FlyAvatarEnv(body=BodylessBodyLayer())
    ctrl = MemoryController()
    return run_episode(env, ctrl, seed=seed, max_steps=_EPISODE_STEPS)


def _run_with_intervention(seed: int, delta: float = 0.5) -> list:
    """Run episode, intervene on hidden state at the midpoint."""
    env = FlyAvatarEnv(body=BodylessBodyLayer())
    ctrl = MemoryController()
    ctrl.reset(seed=seed)
    obs = env.reset(seed=seed)
    transitions = []
    midpoint = _EPISODE_STEPS // 2
    for step in range(_EPISODE_STEPS):
        if step == midpoint:
            # Surgical intervention: shift hidden state.
            ctrl._hidden += delta
        action = ctrl.act(obs)
        t = env.step(action)
        transitions.append(t)
        obs = t.observation
        if t.terminated or t.truncated:
            break
    return transitions


def _run_state_shuffle(seed: int) -> list:
    """Run episode, shuffle hidden state at every step."""
    env = FlyAvatarEnv(body=BodylessBodyLayer())
    ctrl = MemoryController()
    ctrl.reset(seed=seed)
    rng = np.random.default_rng(seed + 9999)
    obs = env.reset(seed=seed)
    transitions = []
    for _ in range(_EPISODE_STEPS):
        rng.shuffle(ctrl._hidden)
        action = ctrl.act(obs)
        t = env.step(action)
        transitions.append(t)
        obs = t.observation
        if t.terminated or t.truncated:
            break
    return transitions


def _run_reactive_baseline(seed: int) -> list:
    """ReducedDescending has no real hidden state — should be ~epiphenomenal."""
    env = FlyAvatarEnv(body=BodylessBodyLayer())
    ctrl = ReducedDescendingController()
    return run_episode(env, ctrl, seed=seed, max_steps=_EPISODE_STEPS)


def run_experiment(output_dir: Path | None = None) -> dict:
    """Run the causal intervention PoC and return results."""
    results = {
        "memory_controller": [],
        "reduced_controller_depth": [],
    }

    for seed in _SEEDS:
        baseline = _run_baseline(seed)
        intervened = _run_with_intervention(seed, delta=0.5)
        shuffled = _run_state_shuffle(seed)

        influence = causal_influence_score(
            baseline, intervened, intervention_magnitude=0.5,
        )
        epiphenomenal = epiphenomenal_test(baseline, shuffled)
        depth = temporal_causal_depth(baseline, max_horizon=8)

        results["memory_controller"].append({
            "seed": seed,
            **influence,
            **epiphenomenal,
            **depth,
            **summarize_metrics(baseline),
        })

        reactive = _run_reactive_baseline(seed)
        reactive_depth = temporal_causal_depth(reactive, max_horizon=8)
        results["reduced_controller_depth"].append({
            "seed": seed,
            **reactive_depth,
        })

    # Aggregate.
    mem_results = results["memory_controller"]
    summary = {
        "mean_causal_influence": float(np.mean([r["causal_influence"] for r in mem_results])),
        "mean_epiphenomenal_score": float(np.mean([r["epiphenomenal_score"] for r in mem_results])),
        "mean_causal_depth": float(np.mean([r["causal_depth"] for r in mem_results])),
        "mean_reactive_depth": float(np.mean(
            [r["causal_depth"] for r in results["reduced_controller_depth"]]
        )),
    }

    # Expectations:
    # - MemoryController should have causal_influence > 0 (state matters)
    # - MemoryController epiphenomenal_score < 1.0 (state is not decorative)
    # - MemoryController should have deeper causal depth than reactive
    summary["state_is_causal"] = summary["mean_causal_influence"] > 0.01
    summary["state_not_epiphenomenal"] = summary["mean_epiphenomenal_score"] < 0.95
    summary["deeper_than_reactive"] = summary["mean_causal_depth"] >= summary["mean_reactive_depth"]

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "causal_intervention_results.json").write_text(
            json.dumps({"summary": summary, "per_seed": results}, indent=2, default=str)
        )

    return {"summary": summary, "per_seed": results}
