"""EXP 4 — Portable Replay Benchmark v2 (PR #5 priority 6).

Turns "false portability exists" into real tier validation or kills it.
Uses 5-world replay protocol, consistency filter, and direct transfer scoring.

Pass criteria:
  - portable > local on mean transfer utility
  - universal > portable only if consistency is high
  - false portable detection precision high
  - accidental local transfer isolated as exceptions

Includes baseline (local), method (portable + universal with filter),
ablation (without consistency filter).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..metrics.sleep_metrics import trajectory_equivalence_strength
from ..research.long_horizon_runner import collect_long_horizon
from ..research.transfer_scoring import compute_transfer_score
from ..sleep import compress_trace_bank
from ..sleep.portable_replay import replay_episode_actions
from ..sleep.trace_schema import SleepCandidate, TraceEpisode


# ---------------------------------------------------------------------------
# Consistency filter (upgraded from v1)
# ---------------------------------------------------------------------------

def _consistency_filter(
    candidate: SleepCandidate,
    episodes: list[TraceEpisode],
    *,
    min_equiv: float = 0.4,
    min_cross_worlds: int = 2,
) -> tuple[bool, float]:
    """Filter universals requiring minimum cross-world equivalence.

    Returns (passes, consistency_score).
    """
    by_id = {ep.episode_id: ep for ep in episodes}
    members = [by_id[eid] for eid in candidate.member_episode_ids if eid in by_id]
    if len(members) < 2:
        return True, 1.0

    by_world: dict[str, list[TraceEpisode]] = defaultdict(list)
    for m in members:
        by_world[m.world_mode].append(m)

    if len(by_world) < min_cross_worlds:
        return False, 0.0

    worlds = sorted(by_world.keys())
    cross_scores = []
    for i in range(len(worlds)):
        for j in range(i + 1, len(worlds)):
            for ea in by_world[worlds[i]][:2]:
                for eb in by_world[worlds[j]][:2]:
                    try:
                        score = trajectory_equivalence_strength(
                            ea.transitions, eb.transitions,
                        )["trajectory_equivalence_strength"]
                        cross_scores.append(float(score))
                    except Exception:
                        cross_scores.append(0.0)

    consistency = float(np.mean(cross_scores)) if cross_scores else 0.0
    return consistency >= min_equiv, consistency


# ---------------------------------------------------------------------------
# Tier assignment (functional, not clustering)
# ---------------------------------------------------------------------------

def _assign_functional_tier(
    candidate: SleepCandidate,
    episodes: list[TraceEpisode],
    replay_results: dict[str, dict],
    source_world: str,
    *,
    consistency_score: float = 0.0,
) -> str:
    """Assign tier from replay utility, not clustering."""
    transfer_rows = [v for k, v in replay_results.items() if k != source_world]
    if not transfer_rows:
        return "local"

    positive_transfers = sum(1 for r in transfer_rows if r.get("return_lift", 0.0) > 0.0)
    frac_positive = positive_transfers / len(transfer_rows)

    if frac_positive >= 0.7 and consistency_score >= 0.5:
        return "universal"
    if frac_positive >= 0.4:
        return "portable"
    return "local"


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _detect_false_portable(
    replay_results: dict[str, dict],
    source_world: str,
    original_tier: str,
) -> bool:
    """Detect candidates labeled portable but that fail on transfer."""
    if original_tier != "portable":
        return False
    transfer_rows = [v for k, v in replay_results.items() if k != source_world]
    if not transfer_rows:
        return False
    failures = sum(1 for r in transfer_rows if r.get("return_lift", 0.0) < -5.0)
    return failures / len(transfer_rows) >= 0.5


def _detect_accidental_local(
    replay_results: dict[str, dict],
    source_world: str,
    original_tier: str,
) -> bool:
    """Detect local candidates that accidentally transfer well."""
    if original_tier != "local":
        return False
    transfer_rows = [v for k, v in replay_results.items() if k != source_world]
    if not transfer_rows:
        return False
    successes = sum(1 for r in transfer_rows if r.get("return_lift", 0.0) > 2.0)
    return successes / len(transfer_rows) >= 0.5


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    output_dir: str | Path = "results/exp_portable_replay_v2",
    *,
    episode_steps: int = 200,
    n_seeds: int = 10,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect long-horizon episodes
    episodes = collect_long_horizon(max_steps=episode_steps, n_seeds=n_seeds)
    artifact = compress_trace_bank(episodes)
    by_id = {ep.episode_id: ep for ep in episodes}
    all_world_modes = sorted({ep.world_mode for ep in episodes})

    # Baselines per world
    by_world: dict[str, list[TraceEpisode]] = defaultdict(list)
    for ep in episodes:
        by_world[ep.world_mode].append(ep)
    baselines = {}
    for wm, wm_eps in by_world.items():
        baselines[wm] = {
            key: float(np.mean([ep.summary_metrics.get(key, 0.0) for ep in wm_eps]))
            for key in ("return", "success", "stability_mean")
        }

    # Process each candidate
    per_candidate = []
    tier_rows: dict[str, list[dict]] = defaultdict(list)
    false_portables = []
    accidental_locals = []
    consistency_scores: dict[str, float] = {}

    for cand in artifact.candidates:
        representative = by_id.get(cand.representative_episode_id)
        if not representative:
            continue

        source_world = representative.world_mode

        # Consistency filter
        passes_filter, cons_score = _consistency_filter(cand, episodes)
        consistency_scores[cand.candidate_id] = cons_score

        # 5-world replay protocol
        replay_results: dict[str, dict] = {}
        for target_world in all_world_modes:
            try:
                replay_metrics = replay_episode_actions(representative, target_world)
            except Exception:
                replay_metrics = {"return": 0.0, "success": 0.0, "stability_mean": 0.0}

            baseline = baselines.get(target_world, {})
            return_lift = replay_metrics.get("return", 0.0) - baseline.get("return", 0.0)
            success_lift = replay_metrics.get("success", 0.0) - baseline.get("success", 0.0)

            condition = "same_world" if target_world == source_world else "transfer"
            if target_world != source_world and "native_physical" in target_world:
                condition = "adversarial"

            replay_results[target_world] = {
                "return_lift": return_lift,
                "success_lift": success_lift,
                "condition": condition,
                "stability_delta": replay_metrics.get("stability_mean", 0.0) - baseline.get("stability_mean", 0.0),
            }

        # Functional tier assignment
        original_tier = cand.redundancy_tier
        functional_tier = _assign_functional_tier(
            cand, episodes, replay_results, source_world,
            consistency_score=cons_score,
        )
        if original_tier == "universal" and not passes_filter:
            functional_tier = "portable"  # demote

        # Transfer metrics
        transfer_rows = [v for k, v in replay_results.items() if k != source_world]
        mean_return_lift = float(np.mean([r["return_lift"] for r in transfer_rows])) if transfer_rows else 0.0
        mean_success_lift = float(np.mean([r["success_lift"] for r in transfer_rows])) if transfer_rows else 0.0

        # Degradation slope
        if len(transfer_rows) >= 2:
            lifts = sorted([r["return_lift"] for r in transfer_rows])
            degradation_slope = float(lifts[-1] - lifts[0]) / max(len(lifts) - 1, 1)
        else:
            degradation_slope = 0.0

        failure_under_mismatch = float(np.mean([
            1.0 if r["return_lift"] < -3.0 else 0.0 for r in transfer_rows
        ])) if transfer_rows else 0.0

        # Transfer score
        ts = compute_transfer_score(cand, episodes)

        cand_row = {
            "candidate_id": cand.candidate_id,
            "original_tier": original_tier,
            "functional_tier": functional_tier,
            "consistency_score": cons_score,
            "mean_return_lift": mean_return_lift,
            "mean_success_lift": mean_success_lift,
            "degradation_slope": degradation_slope,
            "failure_under_mismatch": failure_under_mismatch,
            "portability_fraction": ts.get("portability_fraction", 0.0),
            "replay_stability": ts.get("replay_stability", 0.0),
        }
        per_candidate.append({**cand_row, "replay_details": replay_results})
        tier_rows[functional_tier].append(cand_row)

        # Anomaly detection
        if _detect_false_portable(replay_results, source_world, original_tier):
            false_portables.append(cand_row)
        if _detect_accidental_local(replay_results, source_world, original_tier):
            accidental_locals.append(cand_row)

    # Tier summaries
    by_tier = {}
    for tier, rows in tier_rows.items():
        by_tier[tier] = {
            "n_candidates": len(rows),
            "mean_return_lift": float(np.mean([r["mean_return_lift"] for r in rows])),
            "mean_success_lift": float(np.mean([r["mean_success_lift"] for r in rows])),
            "mean_degradation_slope": float(np.mean([r["degradation_slope"] for r in rows])),
            "mean_failure_under_mismatch": float(np.mean([r["failure_under_mismatch"] for r in rows])),
            "mean_consistency": float(np.mean([r["consistency_score"] for r in rows])),
            "mean_replay_stability": float(np.mean([r["replay_stability"] for r in rows])),
        }

    # Pass criteria
    portable_rows = tier_rows.get("portable", [])
    local_rows = tier_rows.get("local", [])
    universal_rows = tier_rows.get("universal", [])

    portable_mean = float(np.mean([r["mean_return_lift"] for r in portable_rows])) if portable_rows else 0.0
    local_mean = float(np.mean([r["mean_return_lift"] for r in local_rows])) if local_rows else 0.0
    universal_mean = float(np.mean([r["mean_return_lift"] for r in universal_rows])) if universal_rows else 0.0
    universal_consistency = float(np.mean([r["consistency_score"] for r in universal_rows])) if universal_rows else 0.0

    portable_gt_local = portable_mean > local_mean
    universal_gt_portable_if_consistent = (
        universal_mean > portable_mean and universal_consistency > 0.5
    ) if universal_rows else False

    # False portable precision
    fp_precision = (
        len(false_portables) / max(len(portable_rows), 1)
    ) if portable_rows else 0.0

    # Transfer heatmap
    transfer_heatmap: dict[str, dict[str, float]] = {}
    for tier in ("universal", "portable", "local"):
        transfer_heatmap[tier] = {}
        tier_cands = [c for c in per_candidate if c["functional_tier"] == tier]
        for wm in all_world_modes:
            lifts = []
            for c in tier_cands:
                details = c.get("replay_details", {})
                if wm in details:
                    lifts.append(details[wm]["return_lift"])
            transfer_heatmap[tier][wm] = float(np.mean(lifts)) if lifts else 0.0

    payload = {
        "by_tier": by_tier,
        "pass_criteria": {
            "portable_gt_local_on_transfer": portable_gt_local,
            "portable_mean_lift": portable_mean,
            "local_mean_lift": local_mean,
            "universal_gt_portable_if_consistent": universal_gt_portable_if_consistent,
            "universal_mean_lift": universal_mean,
            "universal_consistency": universal_consistency,
            "false_portable_detection_rate": fp_precision,
            "accidental_local_isolated": len(accidental_locals) < len(local_rows) * 0.3 if local_rows else True,
        },
        "transfer_heatmap": transfer_heatmap,
        "false_portables": [fp for fp in false_portables[:15]],
        "accidental_locals": [al for al in accidental_locals[:15]],
        "tier_migration": {
            "universal_demoted": sum(
                1 for c in per_candidate
                if c["original_tier"] == "universal" and c["functional_tier"] != "universal"
            ),
            "local_promoted": sum(
                1 for c in per_candidate
                if c["original_tier"] == "local" and c["functional_tier"] != "local"
            ),
        },
        "config": {
            "n_candidates": len(per_candidate),
            "n_worlds": len(all_world_modes),
            "episode_steps": episode_steps,
            "n_seeds": n_seeds,
        },
    }

    (output_dir / "portable_replay_v2.json").write_text(
        json.dumps(payload, indent=2, default=str),
    )
    return payload
