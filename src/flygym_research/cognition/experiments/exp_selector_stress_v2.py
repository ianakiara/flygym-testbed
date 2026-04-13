"""EXP 5 — Survival vs Scalar Selector Stress Test (PR #5 priority 4).

Forces separation between scalar / survival / basin+collapse selectors
under hard OOD and adversarial candidates.

Pass criteria:
  - survival or basin+collapse gives lower catastrophic rate
  - scalar admits more dangerous candidates
  - selector differences appear clearly under adversarial families

Includes baseline (scalar), method (survival + basin+collapse), ablation (survival only).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..metrics.causal_metrics import temporal_causal_depth
from ..research.candidate_factory import build_candidate_pool
from ..research.long_horizon_runner import collect_long_horizon
from ..sleep import backbone_shared_score, safe_compression_score
from ..sleep.trace_schema import SleepCandidate, TraceEpisode


# ---------------------------------------------------------------------------
# Candidate pool with rich scoring
# ---------------------------------------------------------------------------

def _build_scored_pool(
    episodes: list[TraceEpisode],
    candidates: list[SleepCandidate],
) -> list[dict]:
    pool = []
    by_id = {ep.episode_id: ep for ep in episodes}
    for cand in candidates:
        try:
            bs = backbone_shared_score(cand, episodes)
            sc = safe_compression_score(cand, episodes)
        except Exception:
            continue
        members = [by_id[eid] for eid in cand.member_episode_ids if eid in by_id]
        mean_return = float(np.mean(
            [m.summary_metrics.get("return", 0.0) for m in members],
        )) if members else 0.0

        causal_depths = []
        for m in members[:3]:
            try:
                cd = temporal_causal_depth(m.transitions)
                causal_depths.append(float(cd.get("causal_depth", 0.0)))
            except Exception:
                causal_depths.append(0.0)

        pool.append({
            "candidate_id": cand.candidate_id,
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
            "mean_causal_depth": float(np.mean(causal_depths)) if causal_depths else 0.0,
        })
    return pool


# ---------------------------------------------------------------------------
# Three selectors
# ---------------------------------------------------------------------------

def _scalar_selector(pool: list[dict], *, top_k: int = 10) -> list[dict]:
    weights = {
        "backbone_shared": 0.3,
        "safe_compression": 0.2,
        "mean_return": 0.002,
        "functional_transfer_gain": 0.2,
        "redundancy_score": 0.3,
    }
    for item in pool:
        item["scalar_score"] = sum(weights.get(k, 0.0) * item.get(k, 0.0) for k in weights)
    return sorted(pool, key=lambda x: -x["scalar_score"])[:top_k]


def _survival_selector(
    pool: list[dict],
    *,
    seam_threshold: float = 0.25,
    scale_threshold: float = 0.20,
    interop_threshold: float = 0.25,
    degeneracy_threshold: float = 0.15,
    top_k: int = 10,
) -> list[dict]:
    survivors = [
        item for item in pool
        if item["seam_risk"] <= seam_threshold
        and item["scale_drift"] <= scale_threshold
        and item["interop_loss"] <= interop_threshold
        and item["degeneracy_penalty"] <= degeneracy_threshold
    ]
    for s in survivors:
        s["survival_score"] = s["backbone_shared"] + 0.5 * s["portability_fraction"]
    return sorted(survivors, key=lambda x: -x.get("survival_score", 0.0))[:top_k]


def _basin_collapse_selector(
    pool: list[dict],
    *,
    lambda_collapse: float = 0.5,
    seam_threshold: float = 0.30,
    top_k: int = 10,
) -> list[dict]:
    """Basin size + collapse distance selector."""
    scored = []
    for item in pool:
        basin_size = float(np.clip(
            item["backbone_shared"] + 0.5 * item["portability_fraction"], 0.0, 2.0,
        ))
        collapse_distance = float(np.clip(
            1.0 - item["seam_risk"] - item["scale_drift"] - item["degeneracy_penalty"],
            0.0, 1.0,
        ))
        item["basin_score"] = basin_size + lambda_collapse * collapse_distance
        if item["seam_risk"] <= seam_threshold:
            scored.append(item)
    return sorted(scored, key=lambda x: -x.get("basin_score", 0.0))[:top_k]


# ---------------------------------------------------------------------------
# Kill-test ground truth
# ---------------------------------------------------------------------------

def _is_dangerous(item: dict) -> bool:
    return (
        item["seam_risk"] > 0.35
        or item["scale_drift"] > 0.30
        or item["degeneracy_penalty"] > 0.25
        or item["interop_loss"] > 0.35
    )


def _perturb_pool(pool: list[dict], *, rng: np.random.Generator, noise: float = 0.15) -> list[dict]:
    perturbed = []
    for item in pool:
        p = dict(item)
        for key in ("seam_risk", "interop_loss", "scale_drift", "degeneracy_penalty"):
            p[key] = float(np.clip(item[key] + rng.normal(0, noise), 0.0, 1.0))
        p["backbone_shared"] = float(np.clip(
            item["backbone_shared"] + rng.normal(0, noise * 0.5), -1.0, 1.0,
        ))
        p["mean_return"] = item["mean_return"] + rng.normal(0, 5.0)
        perturbed.append(p)
    return perturbed


def _selection_metrics(selected: list[dict], kill_set: set[str]) -> dict:
    sel_ids = {s["candidate_id"] for s in selected}
    catastrophic = sel_ids & kill_set
    return {
        "n_selected": len(selected),
        "n_dangerous_selected": len(catastrophic),
        "catastrophic_failure_rate": float(len(catastrophic) / max(len(selected), 1)),
        "mean_backbone_shared": float(np.mean([s["backbone_shared"] for s in selected])) if selected else 0.0,
        "mean_seam_risk": float(np.mean([s["seam_risk"] for s in selected])) if selected else 0.0,
        "retained_utility": float(np.mean([s["mean_return"] for s in selected])) if selected else 0.0,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    output_dir: str | Path = "results/exp_selector_stress_v2",
    *,
    episode_steps: int = 100,
    n_seeds: int = 10,
    min_pool_size: int = 250,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # Collect and build pool
    episodes = collect_long_horizon(max_steps=episode_steps, n_seeds=n_seeds)
    all_candidates, family_labels = build_candidate_pool(
        episodes, min_pool_size=min_pool_size, rng=rng,
    )

    pool = _build_scored_pool(episodes, all_candidates)
    if not pool:
        payload = {"error": "Empty pool"}
        (output_dir / "selector_stress_v2.json").write_text(json.dumps(payload, indent=2))
        return payload

    kill_set = {item["candidate_id"] for item in pool if _is_dangerous(item)}

    # In-distribution evaluation
    scalar_sel = _scalar_selector(list(pool))
    survival_sel = _survival_selector(list(pool))
    basin_sel = _basin_collapse_selector(list(pool))

    in_dist = {
        "scalar": _selection_metrics(scalar_sel, kill_set),
        "survival": _selection_metrics(survival_sel, kill_set),
        "basin_collapse": _selection_metrics(basin_sel, kill_set),
    }

    # OOD evaluation (multiple perturbation levels)
    ood_results = {}
    for noise_level in (0.10, 0.20, 0.30):
        ood_pool = _perturb_pool(pool, rng=rng, noise=noise_level)
        kill_ood = {item["candidate_id"] for item in ood_pool if _is_dangerous(item)}
        ood_results[f"noise_{noise_level:.2f}"] = {
            "scalar": _selection_metrics(_scalar_selector(list(ood_pool)), kill_ood),
            "survival": _selection_metrics(_survival_selector(list(ood_pool)), kill_ood),
            "basin_collapse": _selection_metrics(_basin_collapse_selector(list(ood_pool)), kill_ood),
        }

    # Adversarial acceptance rate
    adv_items = [item for item in pool if family_labels.get(item["candidate_id"], "").startswith("adv") or
                 family_labels.get(item["candidate_id"], "") in (
                     "degenerate_converger", "false_portable", "private_only_coherent")]
    adv_ids = {item["candidate_id"] for item in adv_items}
    scalar_adv_accepted = len(adv_ids & {s["candidate_id"] for s in scalar_sel})
    survival_adv_accepted = len(adv_ids & {s["candidate_id"] for s in survival_sel})
    basin_adv_accepted = len(adv_ids & {s["candidate_id"] for s in basin_sel})

    # Pass criteria
    best_ood = ood_results.get("noise_0.20", {})
    surv_cat = best_ood.get("survival", {}).get("catastrophic_failure_rate", 1.0)
    scalar_cat = best_ood.get("scalar", {}).get("catastrophic_failure_rate", 1.0)
    basin_cat = best_ood.get("basin_collapse", {}).get("catastrophic_failure_rate", 1.0)

    payload = {
        "in_distribution": in_dist,
        "out_of_distribution": ood_results,
        "pass_criteria": {
            "survival_lower_catastrophic_than_scalar": surv_cat < scalar_cat,
            "basin_lower_catastrophic_than_scalar": basin_cat < scalar_cat,
            "scalar_admits_more_dangerous": in_dist["scalar"]["n_dangerous_selected"] > in_dist["survival"]["n_dangerous_selected"],
            "clear_separation_under_adversarial": (
                scalar_adv_accepted > survival_adv_accepted
            ),
        },
        "adversarial_acceptance": {
            "scalar": scalar_adv_accepted,
            "survival": survival_adv_accepted,
            "basin_collapse": basin_adv_accepted,
            "n_adversarial_total": len(adv_items),
        },
        "collapse_distances": [
            {
                "candidate_id": item["candidate_id"],
                "collapse_distance": float(np.clip(
                    1.0 - item["seam_risk"] - item["scale_drift"] - item["degeneracy_penalty"],
                    0.0, 1.0,
                )),
                "is_dangerous": _is_dangerous(item),
                "family": family_labels.get(item["candidate_id"], "unknown"),
            }
            for item in pool[:30]
        ],
        "config": {
            "n_candidates": len(pool),
            "n_kill_test": len(kill_set),
            "episode_steps": episode_steps,
            "n_seeds": n_seeds,
        },
    }

    (output_dir / "selector_stress_v2.json").write_text(
        json.dumps(payload, indent=2, default=str),
    )
    return payload
