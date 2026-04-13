"""Experiment 3 — Selection Law (Survival vs Scalar).

Tests:
  ❌ Scalar:  G* = argmin A(G)
  ✅ Survival: G ∈ S (polytope)

Compares scalar regression/logistic selection against survival polytope
with thresholds on composition, scale, interop, memory, closure.
Adds basin size + collapse distance scoring.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..metrics.causal_metrics import temporal_causal_depth
from ..sleep import compress_trace_bank, backbone_shared_score, safe_compression_score
from ..sleep.trace_schema import SleepCandidate, TraceEpisode
from .exp_sleep_trace_compressor import collect_trace_bank


# ---------------------------------------------------------------------------
# Candidate pool generation
# ---------------------------------------------------------------------------

def _build_candidate_pool(
    episodes: list[TraceEpisode],
    candidates: list[SleepCandidate],
) -> list[dict]:
    """Build scored candidate pool with all relevant dimensions."""
    pool = []
    for cand in candidates:
        try:
            bs = backbone_shared_score(cand, episodes)
            sc = safe_compression_score(cand, episodes)
        except Exception:
            continue

        by_id = {ep.episode_id: ep for ep in episodes}
        members = [by_id[eid] for eid in cand.member_episode_ids if eid in by_id]
        mean_return = float(np.mean([m.summary_metrics.get("return", 0.0) for m in members])) if members else 0.0
        mean_success = float(np.mean([m.summary_metrics.get("success", 0.0) for m in members])) if members else 0.0

        # Causal depth
        causal_depths = []
        for m in members[:3]:
            try:
                cd = temporal_causal_depth(m.transitions)
                causal_depths.append(float(cd.get("temporal_causal_depth", 1.0)))
            except Exception:
                causal_depths.append(1.0)
        mean_causal_depth = float(np.mean(causal_depths)) if causal_depths else 1.0

        pool.append({
            "candidate_id": cand.candidate_id,
            "tier": cand.redundancy_tier,
            "backbone_shared": float(bs.get("backbone_shared_score", 0.0)),
            "safe_compression": float(sc.get("safe_compression_score", 0.0)),
            "seam_risk": float(bs.get("seam_risk", 0.0)),
            "interop_loss": float(bs.get("interop_loss", 0.0)),
            "scale_drift": float(bs.get("scale_drift", 0.0)),
            "degeneracy_penalty": float(bs.get("degeneracy_penalty", 0.0)),
            "redundancy_score": float(bs.get("redundancy_score", 0.0)),
            "portability_fraction": float(bs.get("portability_fraction", 0.0)),
            "functional_transfer_gain": float(bs.get("functional_transfer_gain", 0.0)),
            "mean_return": mean_return,
            "mean_success": mean_success,
            "mean_causal_depth": mean_causal_depth,
        })
    return pool


# ---------------------------------------------------------------------------
# Selector A: Scalar (regression/logistic)
# ---------------------------------------------------------------------------

def _scalar_selector(pool: list[dict], *, top_k: int = 5) -> list[dict]:
    """Scalar selector: rank by weighted sum of features."""
    weights = {
        "backbone_shared": 0.3,
        "safe_compression": 0.2,
        "mean_return": 0.002,
        "mean_success": 0.3,
        "functional_transfer_gain": 0.2,
    }
    for item in pool:
        item["scalar_score"] = sum(
            weights.get(k, 0.0) * item.get(k, 0.0) for k in weights
        )
    ranked = sorted(pool, key=lambda x: -x["scalar_score"])
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Selector B: Survival polytope
# ---------------------------------------------------------------------------

def _survival_polytope_selector(
    pool: list[dict],
    *,
    composition_threshold: float = 0.25,
    scale_threshold: float = 0.2,
    interop_threshold: float = 0.25,
    memory_threshold: float = 0.8,
    closure_threshold: float = 0.15,
    top_k: int = 5,
) -> list[dict]:
    """Survival polytope: candidates must pass ALL thresholds to survive."""
    survivors = []
    for item in pool:
        # Composition check
        if item["seam_risk"] > composition_threshold:
            continue
        # Scale check
        if item["scale_drift"] > scale_threshold:
            continue
        # Interop check
        if item["interop_loss"] > interop_threshold:
            continue
        # Memory/causal check
        if item["mean_causal_depth"] < memory_threshold:
            continue
        # Closure check
        if item["degeneracy_penalty"] > closure_threshold:
            continue
        survivors.append(item)

    # Score survivors by basin size + collapse distance
    for s in survivors:
        basin_size = float(np.clip(
            s["backbone_shared"] + 0.5 * s["portability_fraction"],
            0.0, 2.0,
        ))
        collapse_distance = float(np.clip(
            1.0 - s["seam_risk"] - s["scale_drift"] - s["degeneracy_penalty"],
            0.0, 1.0,
        ))
        s["survival_score"] = basin_size + 0.5 * collapse_distance

    ranked = sorted(survivors, key=lambda x: -x.get("survival_score", 0.0))
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Perturbation testing (OOD + catastrophic)
# ---------------------------------------------------------------------------

def _perturb_candidate(item: dict, *, noise_scale: float = 0.15, rng: np.random.Generator | None = None) -> dict:
    """Create a perturbed version of a candidate for OOD testing."""
    rng = rng or np.random.default_rng()
    perturbed = dict(item)
    for key in ["seam_risk", "interop_loss", "scale_drift", "degeneracy_penalty"]:
        perturbed[key] = float(np.clip(
            item[key] + rng.normal(0.0, noise_scale), 0.0, 1.0
        ))
    perturbed["backbone_shared"] = float(np.clip(
        item["backbone_shared"] + rng.normal(0.0, noise_scale * 0.5), -1.0, 1.0
    ))
    perturbed["mean_return"] = item["mean_return"] + rng.normal(0.0, 5.0)
    return perturbed


def _is_kill_test_failure(item: dict) -> bool:
    """Ground truth: should this candidate be killed?"""
    return (
        item["seam_risk"] > 0.35
        or item["scale_drift"] > 0.3
        or item["degeneracy_penalty"] > 0.25
        or item["interop_loss"] > 0.35
    )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    output_dir: str | Path = "results/exp_selection_law",
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect data
    episodes = collect_trace_bank(max_steps=24)
    artifact = compress_trace_bank(episodes)
    pool = _build_candidate_pool(episodes, artifact.candidates)

    if not pool:
        payload = {"error": "No candidates generated", "n_episodes": len(episodes)}
        (output_dir / "selection_law.json").write_text(json.dumps(payload, indent=2))
        return payload

    rng = np.random.default_rng(42)

    # In-distribution evaluation
    scalar_selected = _scalar_selector(list(pool))
    survival_selected = _survival_polytope_selector(list(pool))

    scalar_ids = {s["candidate_id"] for s in scalar_selected}
    survival_ids = {s["candidate_id"] for s in survival_selected}

    # OOD evaluation: perturb all candidates
    ood_pool = [_perturb_candidate(item, rng=rng) for item in pool]
    scalar_ood = _scalar_selector(list(ood_pool))
    survival_ood = _survival_polytope_selector(list(ood_pool))

    # Ground truth kill-test
    kill_set = {item["candidate_id"] for item in pool if _is_kill_test_failure(item)}
    kill_set_ood = {item["candidate_id"] for item in ood_pool if _is_kill_test_failure(item)}

    # Metrics
    def _selection_metrics(selected: list[dict], kill_targets: set[str], label: str) -> dict:
        selected_ids = {s["candidate_id"] for s in selected}
        catastrophic_selected = selected_ids & kill_targets
        return {
            "n_selected": len(selected),
            "n_kill_test_selected": len(catastrophic_selected),
            "catastrophic_failure_rate": float(len(catastrophic_selected) / max(len(selected), 1)),
            "mean_backbone_shared": float(np.mean([s["backbone_shared"] for s in selected])) if selected else 0.0,
            "mean_seam_risk": float(np.mean([s["seam_risk"] for s in selected])) if selected else 0.0,
        }

    in_dist_scalar = _selection_metrics(scalar_selected, kill_set, "scalar_in_dist")
    in_dist_survival = _selection_metrics(survival_selected, kill_set, "survival_in_dist")
    ood_scalar = _selection_metrics(scalar_ood, kill_set_ood, "scalar_ood")
    ood_survival = _selection_metrics(survival_ood, kill_set_ood, "survival_ood")

    # Pass criteria
    survival_better_ood = ood_survival["catastrophic_failure_rate"] < ood_scalar["catastrophic_failure_rate"]
    survival_reduces_catastrophic = ood_survival["catastrophic_failure_rate"] < 0.2
    scalar_misclassifies = ood_scalar["n_kill_test_selected"] > 0

    # Regime classification map
    regime_map = {}
    for item in pool:
        regime = "safe" if not _is_kill_test_failure(item) else "dangerous"
        in_scalar = item["candidate_id"] in scalar_ids
        in_survival = item["candidate_id"] in survival_ids
        regime_map[item["candidate_id"]] = {
            "regime": regime,
            "selected_by_scalar": in_scalar,
            "selected_by_survival": in_survival,
            "seam_risk": item["seam_risk"],
            "backbone_shared": item["backbone_shared"],
        }

    # Basin geometry: distribution of survival scores
    survival_scores = [s.get("survival_score", 0.0) for s in survival_selected]
    scalar_scores = [s.get("scalar_score", 0.0) for s in scalar_selected]

    # Collapse distance graph
    collapse_distances = []
    for item in pool:
        cd = float(np.clip(
            1.0 - item["seam_risk"] - item["scale_drift"] - item["degeneracy_penalty"],
            0.0, 1.0,
        ))
        collapse_distances.append({
            "candidate_id": item["candidate_id"],
            "collapse_distance": cd,
            "is_kill": _is_kill_test_failure(item),
        })

    payload = {
        "in_distribution": {
            "scalar": in_dist_scalar,
            "survival": in_dist_survival,
        },
        "out_of_distribution": {
            "scalar": ood_scalar,
            "survival": ood_survival,
        },
        "pass_criteria": {
            "survival_gt_scalar_ood": survival_better_ood,
            "survival_reduces_catastrophic": survival_reduces_catastrophic,
            "scalar_misclassifies_kill_test": scalar_misclassifies,
        },
        "regime_map_sample": dict(list(regime_map.items())[:15]),
        "collapse_distances": collapse_distances[:20],
        "basin_geometry": {
            "survival_scores": survival_scores,
            "scalar_scores": scalar_scores,
        },
        "n_candidates": len(pool),
        "n_kill_test": len(kill_set),
    }

    (output_dir / "selection_law.json").write_text(json.dumps(payload, indent=2, default=str))
    return payload
