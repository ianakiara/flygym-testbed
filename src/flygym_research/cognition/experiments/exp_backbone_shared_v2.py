"""EXP 2 — BackboneShared Adversarial Benchmark v2 (PR #5 priority 3).

Gives BackboneShared a fair test with 200+ candidates across 7 labeled
families and 9 scoring models.

Pass criteria:
  - BackboneShared AUC ≥ 0.80 (strong: ≥ 0.90)
  - Beats Ω by ≥ 0.08
  - Top-decile precision beats entropy/rank clearly
  - Stable across seeds

Includes baseline (entropy/rank), method (BackboneShared), ablation (Ω variants).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..research.candidate_factory import build_candidate_pool
from ..research.long_horizon_runner import collect_long_horizon
from ..research.transfer_scoring import compute_transfer_score
from ..sleep import backbone_shared_score


# ---------------------------------------------------------------------------
# 9 scoring models
# ---------------------------------------------------------------------------

def _score_entropy(candidate, episodes):
    by_id = {ep.episode_id: ep for ep in episodes}
    members = [by_id[eid] for eid in candidate.member_episode_ids if eid in by_id]
    if not members:
        return 0.0
    returns = [m.summary_metrics.get("return", 0.0) for m in members]
    if len(returns) < 2 or np.std(returns) < 1e-8:
        return 0.5
    normalized = (np.array(returns) - np.min(returns)) / (np.max(returns) - np.min(returns) + 1e-8)
    p = normalized / (normalized.sum() + 1e-8)
    p = p[p > 0]
    entropy = -float(np.sum(p * np.log(p + 1e-12)))
    return float(np.clip(entropy / np.log(max(len(returns), 2)), 0.0, 1.0))


def _score_rank(candidate, episodes):
    eq = float(candidate.score_components.get("mean_equivalence_strength", 0.0))
    size = float(candidate.score_components.get("cluster_size", 1.0))
    return float(np.clip(0.5 * eq + 0.5 * np.tanh(size / 5.0), 0.0, 1.0))


def _score_omega(candidate, episodes):
    result = backbone_shared_score(candidate, episodes)
    return float(result.get("redundancy_score", 0.0))


def _score_omega_shared(candidate, episodes):
    result = backbone_shared_score(candidate, episodes)
    return float(
        result.get("redundancy_score", 0.0) * 0.6
        + result.get("portability_fraction", 0.0) * 0.4
    )


def _score_backbone_shared(candidate, episodes):
    result = backbone_shared_score(candidate, episodes)
    return float(result.get("backbone_shared_score", 0.0))


def _score_return_only(candidate, episodes):
    by_id = {ep.episode_id: ep for ep in episodes}
    members = [by_id[eid] for eid in candidate.member_episode_ids if eid in by_id]
    if not members:
        return 0.0
    mean_ret = float(np.mean([m.summary_metrics.get("return", 0.0) for m in members]))
    return float(np.clip((mean_ret + 100) / 200, 0.0, 1.0))


def _score_low_drift_only(candidate, episodes):
    drift = float(candidate.score_components.get("scale_drift", 0.5))
    return float(np.clip(1.0 - drift, 0.0, 1.0))


def _score_transfer_only(candidate, episodes):
    ts = compute_transfer_score(candidate, episodes)
    score = (
        ts.get("functional_transfer_gain", 0.0)
        + 0.5 * ts.get("portability_fraction", 0.0)
    )
    return float(np.clip(score, 0.0, 1.0))


def _score_combined_learned(candidate, episodes):
    """Simulated learned model: weighted combination of all sub-scores."""
    bs = backbone_shared_score(candidate, episodes)
    ts = compute_transfer_score(candidate, episodes)
    return float(
        0.25 * bs.get("backbone_shared_score", 0.0)
        + 0.15 * bs.get("redundancy_score", 0.0)
        + 0.15 * (1.0 - bs.get("seam_risk", 0.5))
        + 0.15 * (1.0 - bs.get("interop_loss", 0.5))
        + 0.10 * ts.get("portability_fraction", 0.0)
        + 0.10 * (1.0 - bs.get("scale_drift", 0.5))
        + 0.10 * ts.get("replay_stability", 0.0)
    )


SCORING_MODELS = {
    "entropy": _score_entropy,
    "rank": _score_rank,
    "omega": _score_omega,
    "omega_shared": _score_omega_shared,
    "backbone_shared": _score_backbone_shared,
    "return_only": _score_return_only,
    "low_drift_only": _score_low_drift_only,
    "transfer_only": _score_transfer_only,
    "combined_learned": _score_combined_learned,
}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _compute_auc(scores: list[float], labels: list[bool]) -> float:
    if not scores or len(scores) != len(labels):
        return 0.5
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    tp = 0
    auc = 0.0
    for _score, label in paired:
        if label:
            tp += 1
        else:
            auc += tp
    return float(auc / (n_pos * n_neg))


def _compute_auprc(scores: list[float], labels: list[bool]) -> float:
    if not scores or sum(labels) == 0:
        return 0.0
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    tp_cum = 0
    precision_sum = 0.0
    for i, (_s, lbl) in enumerate(paired):
        if lbl:
            tp_cum += 1
            precision_sum += tp_cum / (i + 1)
    return float(precision_sum / max(sum(labels), 1))


def _precision_at_top(scores: list[float], labels: list[bool], k: int = 10) -> float:
    if not scores or k <= 0:
        return 0.0
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    top_k = paired[:min(k, len(paired))]
    return float(np.mean([float(label) for _, label in top_k]))


def _compute_seed_stability(
    episodes: list,
    *,
    min_pool_size: int,
) -> dict[str, object]:
    """Estimate BackboneShared AUC stability over per-seed candidate pools."""
    per_seed_auc: dict[int, float] = {}
    for seed in sorted({ep.seed for ep in episodes}):
        seed_episodes = [ep for ep in episodes if ep.seed == seed]
        if len(seed_episodes) < 2:
            continue
        seed_candidates, seed_labels = build_candidate_pool(
            seed_episodes,
            min_pool_size=min_pool_size,
        )
        if not seed_candidates:
            continue
        scores = []
        labels = []
        for cand in seed_candidates:
            scores.append(_score_backbone_shared(cand, seed_episodes))
            labels.append(seed_labels.get(cand.candidate_id) == "valid_shared")
        per_seed_auc[seed] = _compute_auc(scores, labels)

    aucs = list(per_seed_auc.values())
    if len(aucs) < 2:
        return {
            "per_seed_auc": per_seed_auc,
            "mean_auc": float(np.mean(aucs)) if aucs else 0.0,
            "std_auc": 0.0,
            "cv_auc": 0.0,
            "stable": False,
        }

    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))
    cv_auc = float(std_auc / (abs(mean_auc) + 1e-8))
    return {
        "per_seed_auc": per_seed_auc,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "cv_auc": cv_auc,
        "stable": cv_auc < 0.10,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    output_dir: str | Path = "results/exp_backbone_shared_v2",
    *,
    episode_steps: int = 100,
    n_seeds: int = 10,
    min_pool_size: int = 200,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect long-horizon episodes
    episodes = collect_long_horizon(max_steps=episode_steps, n_seeds=n_seeds)

    # Build large labeled candidate pool
    all_candidates, family_labels = build_candidate_pool(
        episodes, min_pool_size=min_pool_size,
    )

    # Score all candidates with all 9 models
    scores_matrix: dict[str, dict[str, float]] = {m: {} for m in SCORING_MODELS}
    for cand in all_candidates:
        for model_name, score_fn in SCORING_MODELS.items():
            try:
                scores_matrix[model_name][cand.candidate_id] = score_fn(cand, episodes)
            except Exception:
                scores_matrix[model_name][cand.candidate_id] = 0.0

    # Binary classification: valid_shared vs everything else
    is_valid = {cid: label == "valid_shared" for cid, label in family_labels.items()}

    # Evaluate each model
    model_results = {}
    for model_name in SCORING_MODELS:
        cand_ids = [c.candidate_id for c in all_candidates]
        model_scores = [scores_matrix[model_name].get(cid, 0.0) for cid in cand_ids]
        model_labels = [is_valid.get(cid, False) for cid in cand_ids]

        auc = _compute_auc(model_scores, model_labels)
        auprc = _compute_auprc(model_scores, model_labels)
        p_at_10 = _precision_at_top(model_scores, model_labels, k=10)
        p_at_top_decile = _precision_at_top(
            model_scores, model_labels, k=max(1, len(cand_ids) // 10),
        )

        # Robustness: adversarial vs non-adversarial gap
        adv_scores = [
            scores_matrix[model_name].get(cid, 0.0)
            for cid, lbl in family_labels.items()
            if lbl not in ("valid_shared", "ambiguous")
        ]
        valid_scores = [
            scores_matrix[model_name].get(cid, 0.0)
            for cid, lbl in family_labels.items()
            if lbl == "valid_shared"
        ]
        adv_mean = float(np.mean(adv_scores)) if adv_scores else 0.0
        valid_mean = float(np.mean(valid_scores)) if valid_scores else 0.0

        model_results[model_name] = {
            "auc": auc,
            "auprc": auprc,
            "precision_at_10": p_at_10,
            "precision_at_top_decile": p_at_top_decile,
            "mean_valid_score": valid_mean,
            "mean_adversarial_score": adv_mean,
            "rejection_gap": valid_mean - adv_mean,
        }

    # Confusion matrix across families
    confusion = {}
    for family in sorted(set(family_labels.values())):
        family_cids = [cid for cid, lbl in family_labels.items() if lbl == family]
        if not family_cids:
            continue
        confusion[family] = {"count": len(family_cids)}
        for model_name in SCORING_MODELS:
            fam_scores = [scores_matrix[model_name].get(cid, 0.0) for cid in family_cids]
            confusion[family][f"{model_name}_mean"] = float(np.mean(fam_scores))
            confusion[family][f"{model_name}_std"] = float(np.std(fam_scores))

    # Feature importance (ablation): measure penalty contributions
    ablation_results = {}
    real_cands = [c for c in all_candidates if not c.candidate_id.startswith("adv-")][:20]
    for cand in real_cands:
        try:
            full = backbone_shared_score(cand, episodes)
        except Exception:
            continue
        ablation_results[cand.candidate_id] = {
            "full_score": float(full.get("backbone_shared_score", 0.0)),
            "redundancy_score": float(full.get("redundancy_score", 0.0)),
            "seam_risk": float(full.get("seam_risk", 0.0)),
            "interop_loss": float(full.get("interop_loss", 0.0)),
            "scale_drift": float(full.get("scale_drift", 0.0)),
            "degeneracy_penalty": float(full.get("degeneracy_penalty", 0.0)),
        }

    # Pass criteria
    bs_auc = model_results["backbone_shared"]["auc"]
    omega_auc = model_results["omega"]["auc"]
    entropy_p10 = model_results["entropy"]["precision_at_top_decile"]
    rank_p10 = model_results["rank"]["precision_at_top_decile"]
    bs_p10 = model_results["backbone_shared"]["precision_at_top_decile"]
    seed_stability = _compute_seed_stability(episodes, min_pool_size=max(20, min_pool_size // 2))

    payload = {
        "model_results": model_results,
        "pass_criteria": {
            "backbone_shared_auc_ge_0.80": bs_auc >= 0.80,
            "backbone_shared_auc_ge_0.90_strong": bs_auc >= 0.90,
            "backbone_shared_auc": bs_auc,
            "beats_omega_by_0.08": (bs_auc - omega_auc) >= 0.08,
            "bs_minus_omega_auc": bs_auc - omega_auc,
            "bs_top_decile_beats_entropy": bs_p10 > entropy_p10,
            "bs_top_decile_beats_rank": bs_p10 > rank_p10,
            "stable_across_seeds": bool(seed_stability["stable"]),
        },
        "confusion_matrix": confusion,
        "feature_importance": ablation_results,
        "seed_stability": seed_stability,
        "failure_examples": [
            {
                "candidate_id": cid,
                "family": family_labels.get(cid, "unknown"),
                "backbone_shared_score": scores_matrix["backbone_shared"].get(cid, 0.0),
            }
            for cid in list(family_labels.keys())[:20]
            if (
                (family_labels.get(cid) == "valid_shared"
                 and scores_matrix["backbone_shared"].get(cid, 0.0) < 0.1)
                or (family_labels.get(cid) != "valid_shared"
                    and scores_matrix["backbone_shared"].get(cid, 0.0) > 0.3)
            )
        ][:20],
        "family_distribution": {
            family: sum(1 for lbl in family_labels.values() if lbl == family)
            for family in sorted(set(family_labels.values()))
        },
        "config": {
            "n_candidates": len(all_candidates),
            "n_real_candidates": sum(
                1 for c in all_candidates if not c.candidate_id.startswith("adv-")
            ),
            "n_synthetic": sum(
                1 for c in all_candidates if c.candidate_id.startswith("adv-")
            ),
            "episode_steps": episode_steps,
            "n_seeds": n_seeds,
        },
    }

    (output_dir / "backbone_shared_v2.json").write_text(
        json.dumps(payload, indent=2, default=str),
    )
    return payload
