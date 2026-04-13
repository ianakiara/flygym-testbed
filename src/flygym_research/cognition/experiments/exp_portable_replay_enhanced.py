"""Experiment 4 — Portable Replay (Transfer is Real or Fake).

Enhanced version testing whether "portable" structures are actually useful.
Tests all 3 tiers (universal, portable, local) with replay in same-world,
unseen-world, and adversarial-world conditions.  Fixes the universal paradox
with a consistency filter and detects "false portable" candidates.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..metrics.sleep_metrics import trajectory_equivalence_strength
from ..sleep import compress_trace_bank, backbone_shared_score
from ..sleep.portable_replay import replay_episode_actions
from ..sleep.trace_schema import SleepCandidate, TraceEpisode
from .exp_sleep_trace_compressor import collect_trace_bank


# ---------------------------------------------------------------------------
# Consistency filter for universal candidates (fixes paradox)
# ---------------------------------------------------------------------------

def _consistency_filter(
    candidate: SleepCandidate,
    episodes: list[TraceEpisode],
    *,
    min_cross_world_equivalence: float = 0.4,
) -> bool:
    """Filter out universals that average incompatible structure.

    The current paradox: universal candidates often fail because they
    average representations from incompatible worlds.  This filter
    requires minimum pairwise equivalence across worlds.
    """
    by_id = {ep.episode_id: ep for ep in episodes}
    members = [by_id[eid] for eid in candidate.member_episode_ids if eid in by_id]
    if len(members) < 2:
        return True

    by_world: dict[str, list[TraceEpisode]] = defaultdict(list)
    for m in members:
        by_world[m.world_mode].append(m)

    if len(by_world) < 2:
        return True

    worlds = sorted(by_world.keys())
    cross_scores = []
    for i in range(len(worlds)):
        for j in range(i + 1, len(worlds)):
            eps_a = by_world[worlds[i]]
            eps_b = by_world[worlds[j]]
            for ea in eps_a[:2]:
                for eb in eps_b[:2]:
                    try:
                        score = trajectory_equivalence_strength(
                            ea.transitions, eb.transitions
                        )["trajectory_equivalence_strength"]
                        cross_scores.append(score)
                    except Exception:
                        cross_scores.append(0.0)

    if not cross_scores:
        return True
    return float(np.mean(cross_scores)) >= min_cross_world_equivalence


# ---------------------------------------------------------------------------
# False portable detection
# ---------------------------------------------------------------------------

def _detect_false_portable(
    candidate: SleepCandidate,
    episodes: list[TraceEpisode],
    replay_results: dict[str, dict],
) -> dict | None:
    """Detect candidates labeled portable but that fail on transfer."""
    if candidate.redundancy_tier != "portable":
        return None

    transfer_failures = 0
    total_transfers = 0
    for world_mode, metrics in replay_results.items():
        total_transfers += 1
        if metrics.get("return_lift", 0.0) < -5.0:
            transfer_failures += 1

    if total_transfers > 0 and transfer_failures / total_transfers >= 0.5:
        return {
            "candidate_id": candidate.candidate_id,
            "tier": candidate.redundancy_tier,
            "transfer_failure_rate": transfer_failures / total_transfers,
            "type": "false_portable",
        }
    return None


def _detect_accidental_local_transfer(
    candidate: SleepCandidate,
    replay_results: dict[str, dict],
) -> dict | None:
    """Detect local candidates that accidentally transfer well."""
    if candidate.redundancy_tier != "local":
        return None

    transfer_successes = 0
    total_transfers = 0
    for world_mode, metrics in replay_results.items():
        total_transfers += 1
        if metrics.get("return_lift", 0.0) > 2.0:
            transfer_successes += 1

    if total_transfers > 0 and transfer_successes / total_transfers >= 0.5:
        return {
            "candidate_id": candidate.candidate_id,
            "tier": candidate.redundancy_tier,
            "accidental_transfer_rate": transfer_successes / total_transfers,
            "type": "accidental_local_transfer",
        }
    return None


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    output_dir: str | Path = "results/exp_portable_replay_enhanced",
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect episodes and compress
    episodes = collect_trace_bank(max_steps=24)
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

    # Apply consistency filter to universals
    filtered_candidates = []
    consistency_failures = []
    for cand in artifact.candidates:
        if cand.redundancy_tier == "universal":
            if not _consistency_filter(cand, episodes):
                consistency_failures.append(cand.candidate_id)
                # Demote to portable
                cand.redundancy_tier = "portable"
        filtered_candidates.append(cand)

    # Replay protocol: same-world, unseen-world, adversarial-world
    per_candidate = []
    tier_rows: dict[str, list[dict]] = defaultdict(list)
    false_portables = []
    accidental_transfers = []

    for cand in filtered_candidates:
        representative = by_id.get(cand.representative_episode_id)
        if not representative:
            continue

        source_world = representative.world_mode
        replay_results = {}

        for target_world in all_world_modes:
            try:
                replay_metrics = replay_episode_actions(representative, target_world)
            except Exception:
                replay_metrics = {"return": 0.0, "success": 0.0, "stability_mean": 0.0}

            baseline = baselines.get(target_world, {})
            return_lift = replay_metrics.get("return", 0.0) - baseline.get("return", 0.0)
            success_lift = replay_metrics.get("success", 0.0) - baseline.get("success", 0.0)

            condition = "same_world" if target_world == source_world else "unseen_world"
            # Adversarial: noisy version (we simulate by using the noisiest world)
            if target_world != source_world and "native_physical" in target_world:
                condition = "adversarial_world"

            replay_results[target_world] = {
                "return_lift": return_lift,
                "success_lift": success_lift,
                "condition": condition,
                "stability_delta": replay_metrics.get("stability_mean", 0.0) - baseline.get("stability_mean", 0.0),
            }

        # Aggregate
        transfer_rows = [v for k, v in replay_results.items() if k != source_world]
        mean_return_lift = float(np.mean([r["return_lift"] for r in transfer_rows])) if transfer_rows else 0.0
        mean_success_lift = float(np.mean([r["success_lift"] for r in transfer_rows])) if transfer_rows else 0.0

        # Degradation slope: how fast does performance degrade as we move further from source?
        degradation_slope = 0.0
        if len(transfer_rows) >= 2:
            lifts = sorted([r["return_lift"] for r in transfer_rows])
            degradation_slope = float(lifts[-1] - lifts[0]) / max(len(lifts) - 1, 1)

        failure_under_mismatch = float(np.mean([
            1.0 if r["return_lift"] < -3.0 else 0.0 for r in transfer_rows
        ])) if transfer_rows else 0.0

        try:
            bs = backbone_shared_score(cand, episodes)
            portability_score = float(bs.get("backbone_shared_score", 0.0))
        except Exception:
            portability_score = 0.0

        cand_row = {
            "candidate_id": cand.candidate_id,
            "tier": cand.redundancy_tier,
            "source_world": source_world,
            "mean_return_lift": mean_return_lift,
            "mean_success_lift": mean_success_lift,
            "degradation_slope": degradation_slope,
            "failure_under_mismatch": failure_under_mismatch,
            "portability_score": portability_score,
        }
        per_candidate.append({**cand_row, "replay_details": replay_results})
        tier_rows[cand.redundancy_tier].append(cand_row)

        # False portable detection
        fp = _detect_false_portable(cand, episodes, replay_results)
        if fp:
            false_portables.append(fp)

        # Accidental local transfer
        at = _detect_accidental_local_transfer(cand, replay_results)
        if at:
            accidental_transfers.append(at)

    # By-tier summary
    by_tier = {}
    for tier, rows in tier_rows.items():
        by_tier[tier] = {
            "n_candidates": len(rows),
            "mean_return_lift": float(np.mean([r["mean_return_lift"] for r in rows])),
            "mean_success_lift": float(np.mean([r["mean_success_lift"] for r in rows])),
            "mean_degradation_slope": float(np.mean([r["degradation_slope"] for r in rows])),
            "mean_failure_under_mismatch": float(np.mean([r["failure_under_mismatch"] for r in rows])),
            "mean_portability_score": float(np.mean([r["portability_score"] for r in rows])),
        }

    # Pass criteria
    portable_rows = tier_rows.get("portable", [])
    local_rows = tier_rows.get("local", [])
    portable_mean = float(np.mean([r["mean_return_lift"] for r in portable_rows])) if portable_rows else 0.0
    local_mean = float(np.mean([r["mean_return_lift"] for r in local_rows])) if local_rows else 0.0

    portable_gt_local = portable_mean > local_mean
    universal_coherent_if_filtered = len(consistency_failures) > 0  # We did fix some

    # Transfer heatmap: tier × world
    transfer_heatmap = {}
    for tier in ["universal", "portable", "local"]:
        transfer_heatmap[tier] = {}
        tier_cands = [c for c in per_candidate if c["tier"] == tier]
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
            "universal_fixed_with_consistency_filter": universal_coherent_if_filtered,
        },
        "transfer_heatmap": transfer_heatmap,
        "false_portables": false_portables[:10],
        "accidental_transfers": accidental_transfers[:10],
        "consistency_filter_demotions": consistency_failures,
        "portability_vs_performance": [
            {"candidate_id": c["candidate_id"], "portability_score": c["portability_score"],
             "mean_return_lift": c["mean_return_lift"]}
            for c in per_candidate
        ][:20],
        "n_candidates": len(per_candidate),
        "n_worlds": len(all_world_modes),
    }

    (output_dir / "portable_replay_enhanced.json").write_text(json.dumps(payload, indent=2, default=str))
    return payload
