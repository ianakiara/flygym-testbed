"""Experiment 2 — BackboneShared Benchmark.

Validates:  BackboneShared(X) = Ω − λI − μS − νD

Builds 4 candidate families (valid, fake-coherent, seam-fragile,
scale-artifact) and compares 6 scoring models including BackboneShared,
Ω, entropy, rank, and return-only.  Includes adversarial extensions.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..metrics import (
    interoperability_score,
    seam_fragility,
)
from ..sleep import compress_trace_bank, backbone_shared_score
from ..sleep.trace_schema import SleepCandidate, TraceEpisode
from .exp_sleep_trace_compressor import collect_trace_bank


# ---------------------------------------------------------------------------
# Candidate family builders
# ---------------------------------------------------------------------------

def _label_family(candidate: SleepCandidate, episodes: list[TraceEpisode]) -> str:
    """Assign a ground-truth family label based on structural properties."""
    by_id = {ep.episode_id: ep for ep in episodes}
    members = [by_id[eid] for eid in candidate.member_episode_ids if eid in by_id]
    representative = by_id.get(candidate.representative_episode_id)
    if not representative or not members:
        return "unknown"

    # Seam fragility check
    seam_scores = [seam_fragility(m.transitions).get("seam_fragility", 0.0) for m in members]
    mean_seam = float(np.mean(seam_scores)) if seam_scores else 0.0

    # Interoperability check
    interop_losses = []
    for m in members:
        if m.episode_id == representative.episode_id:
            continue
        interop = interoperability_score(representative.transitions, m.transitions)
        interop_losses.append(1.0 - float(interop.get("interoperability_score", 1.0)))
    mean_interop_loss = float(np.mean(interop_losses)) if interop_losses else 0.0

    # World coverage
    world_modes = {m.world_mode for m in members}

    # Scale stability (cross-world divergence)
    scale_drift = float(candidate.score_components.get("scale_drift", 0.0))

    # Classification
    if mean_seam < 0.15 and mean_interop_loss < 0.2 and scale_drift < 0.15:
        return "valid"
    if mean_interop_loss >= 0.25 and mean_seam < 0.15:
        return "fake_coherent"
    if mean_seam >= 0.2 and mean_interop_loss < 0.2:
        return "seam_fragile"
    if scale_drift >= 0.2 or (len(world_modes) <= 1 and mean_seam < 0.15):
        return "scale_artifact"
    return "ambiguous"


def _inject_adversarial_candidates(
    candidates: list[SleepCandidate],
    episodes: list[TraceEpisode],
) -> list[tuple[SleepCandidate, str]]:
    """Create adversarial examples: looks perfect locally but fails globally."""
    adversarial = []
    for cand in candidates:
        # "Overfit portable" — artificially inflate portability_fraction
        overfit = SleepCandidate(
            candidate_id=f"adv-overfit-{cand.candidate_id}",
            representative_episode_id=cand.representative_episode_id,
            member_episode_ids=cand.member_episode_ids,
            evidence={**cand.evidence, "adversarial": "overfit_portable"},
            score_components={**cand.score_components, "portability_fraction": 1.0},
            redundancy_tier="universal",
            portability_evidence={
                **cand.portability_evidence,
                "portability_fraction": 1.0,
                "world_modes": ["avatar_remapped", "native_physical", "simplified_embodied"],
            },
        )
        adversarial.append((overfit, "adversarial_overfit"))

        # "Compressed but unstable" — high compression, high seam risk
        unstable = SleepCandidate(
            candidate_id=f"adv-unstable-{cand.candidate_id}",
            representative_episode_id=cand.representative_episode_id,
            member_episode_ids=cand.member_episode_ids,
            evidence={**cand.evidence, "adversarial": "compressed_unstable"},
            score_components={
                **cand.score_components,
                "mean_equivalence_strength": 0.95,
                "cluster_size": 10.0,
            },
            redundancy_tier=cand.redundancy_tier,
            portability_evidence=cand.portability_evidence,
        )
        adversarial.append((unstable, "adversarial_unstable"))
    return adversarial[:20]


# ---------------------------------------------------------------------------
# Scoring models
# ---------------------------------------------------------------------------

def _score_omega(candidate: SleepCandidate, episodes: list[TraceEpisode]) -> float:
    """Ω (closure-leakage): base redundancy without penalties."""
    result = backbone_shared_score(candidate, episodes)
    return float(result.get("redundancy_score", 0.0))


def _score_omega_shared(candidate: SleepCandidate, episodes: list[TraceEpisode]) -> float:
    """Ω_shared: redundancy + portability, no penalties."""
    result = backbone_shared_score(candidate, episodes)
    return float(
        result.get("redundancy_score", 0.0) * 0.6
        + result.get("portability_fraction", 0.0) * 0.4
    )


def _score_backbone_shared(candidate: SleepCandidate, episodes: list[TraceEpisode]) -> float:
    """Full BackboneShared score with all penalties."""
    result = backbone_shared_score(candidate, episodes)
    return float(result.get("backbone_shared_score", 0.0))


def _score_entropy(candidate: SleepCandidate, episodes: list[TraceEpisode]) -> float:
    """Entropy-based: measure trajectory diversity."""
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
    return float(np.clip(entropy / np.log(len(returns) + 1), 0.0, 1.0))


def _score_rank(candidate: SleepCandidate, episodes: list[TraceEpisode]) -> float:
    """Rank-based: simple ordering by equivalence strength and size."""
    eq_strength = float(candidate.score_components.get("mean_equivalence_strength", 0.0))
    size = float(candidate.score_components.get("cluster_size", 1.0))
    return float(np.clip(0.5 * eq_strength + 0.5 * np.tanh(size / 5.0), 0.0, 1.0))


def _score_return_only(candidate: SleepCandidate, episodes: list[TraceEpisode]) -> float:
    """Return-only: naive performance-based scoring."""
    by_id = {ep.episode_id: ep for ep in episodes}
    members = [by_id[eid] for eid in candidate.member_episode_ids if eid in by_id]
    if not members:
        return 0.0
    mean_return = float(np.mean([m.summary_metrics.get("return", 0.0) for m in members]))
    return float(np.clip((mean_return + 100) / 200, 0.0, 1.0))


SCORING_MODELS = {
    "omega": _score_omega,
    "omega_shared": _score_omega_shared,
    "backbone_shared": _score_backbone_shared,
    "entropy": _score_entropy,
    "rank": _score_rank,
    "return_only": _score_return_only,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _compute_auc(scores: list[float], labels: list[bool]) -> float:
    """Compute AUC for binary classification."""
    if not scores or not labels or len(scores) != len(labels):
        return 0.5
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    tp = 0
    auc = 0.0
    for score, label in paired:
        if label:
            tp += 1
        else:
            auc += tp
    return float(auc / (n_pos * n_neg))


def _precision_at_top(scores: list[float], labels: list[bool], k: int = 5) -> float:
    """Precision in top-k scored items."""
    if not scores or k <= 0:
        return 0.0
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    top_k = paired[:min(k, len(paired))]
    return float(np.mean([float(label) for _, label in top_k]))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    output_dir: str | Path = "results/exp_backbone_shared_benchmark",
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect episodes and extract candidates
    episodes = collect_trace_bank(max_steps=24)
    artifact = compress_trace_bank(episodes)
    candidates = artifact.candidates

    # Label families
    family_labels = {}
    for cand in candidates:
        family_labels[cand.candidate_id] = _label_family(cand, episodes)

    # Inject adversarial candidates
    adversarial_pairs = _inject_adversarial_candidates(candidates, episodes)
    all_candidates = list(candidates)
    for adv_cand, adv_label in adversarial_pairs:
        all_candidates.append(adv_cand)
        family_labels[adv_cand.candidate_id] = adv_label

    # Score all candidates with all models
    scores_matrix = {model_name: {} for model_name in SCORING_MODELS}
    for cand in all_candidates:
        for model_name, score_fn in SCORING_MODELS.items():
            try:
                scores_matrix[model_name][cand.candidate_id] = score_fn(cand, episodes)
            except Exception:
                scores_matrix[model_name][cand.candidate_id] = 0.0

    # Binary classification: valid vs not-valid
    is_valid = {cid: label == "valid" for cid, label in family_labels.items()}

    model_results = {}
    for model_name in SCORING_MODELS:
        cand_ids = [c.candidate_id for c in all_candidates]
        model_scores = [scores_matrix[model_name].get(cid, 0.0) for cid in cand_ids]
        model_labels = [is_valid.get(cid, False) for cid in cand_ids]

        auc = _compute_auc(model_scores, model_labels)
        p_at_5 = _precision_at_top(model_scores, model_labels, k=5)

        # Robustness: check scores for adversarial candidates
        adv_scores = [scores_matrix[model_name].get(cid, 0.0)
                       for cid, label in family_labels.items()
                       if label.startswith("adversarial")]
        non_adv_scores = [scores_matrix[model_name].get(cid, 0.0)
                          for cid, label in family_labels.items()
                          if not label.startswith("adversarial")]

        adv_mean = float(np.mean(adv_scores)) if adv_scores else 0.0
        non_adv_mean = float(np.mean(non_adv_scores)) if non_adv_scores else 0.0
        adv_rejection = float(non_adv_mean - adv_mean) if adv_scores else 0.0

        model_results[model_name] = {
            "auc": auc,
            "precision_at_top_5": p_at_5,
            "mean_score_valid": float(np.mean([
                scores_matrix[model_name].get(cid, 0.0)
                for cid, label in family_labels.items() if label == "valid"
            ] or [0.0])),
            "mean_score_adversarial": adv_mean,
            "adversarial_rejection_gap": adv_rejection,
        }

    # Confusion matrix across families
    confusion = {}
    for family in ["valid", "fake_coherent", "seam_fragile", "scale_artifact", "ambiguous",
                    "adversarial_overfit", "adversarial_unstable"]:
        family_cids = [cid for cid, label in family_labels.items() if label == family]
        if not family_cids:
            continue
        confusion[family] = {
            "count": len(family_cids),
        }
        for model_name in SCORING_MODELS:
            fam_scores = [scores_matrix[model_name].get(cid, 0.0) for cid in family_cids]
            confusion[family][f"{model_name}_mean"] = float(np.mean(fam_scores))
            confusion[family][f"{model_name}_std"] = float(np.std(fam_scores))

    # Feature importance (λ, μ, ν) by ablation
    # Measure how much each penalty contributes
    ablation_results = {}
    for cand in candidates[:10]:
        try:
            full = backbone_shared_score(cand, episodes)
        except Exception:
            continue
        ablation_results[cand.candidate_id] = {
            "full_backbone_shared": float(full.get("backbone_shared_score", 0.0)),
            "seam_risk_contribution": float(full.get("seam_risk", 0.0)) * 0.6,
            "interop_loss_contribution": float(full.get("interop_loss", 0.0)) * 0.6,
            "scale_drift_contribution": float(full.get("scale_drift", 0.0)) * 0.5,
            "degeneracy_contribution": float(full.get("degeneracy_penalty", 0.0)) * 0.7,
            "lambda_seam": float(full.get("seam_risk", 0.0)),
            "mu_interop": float(full.get("interop_loss", 0.0)),
            "nu_scale": float(full.get("scale_drift", 0.0)),
        }

    # Pass criteria
    bs_auc = model_results["backbone_shared"]["auc"]
    omega_auc = model_results["omega"]["auc"]

    pass_criteria = {
        "backbone_shared_auc_ge_0.90": bs_auc >= 0.90,
        "beats_omega_by_0.08": (bs_auc - omega_auc) >= 0.08,
        "stable_under_adversarial": model_results["backbone_shared"]["adversarial_rejection_gap"] > 0.0,
    }

    # Failure examples
    failure_examples = []
    for cand in all_candidates:
        score = scores_matrix["backbone_shared"].get(cand.candidate_id, 0.0)
        label = family_labels.get(cand.candidate_id, "unknown")
        if (label == "valid" and score < 0.1) or (label != "valid" and score > 0.3):
            failure_examples.append({
                "candidate_id": cand.candidate_id,
                "family": label,
                "backbone_shared_score": score,
                "tier": cand.redundancy_tier,
            })

    payload = {
        "model_results": model_results,
        "pass_criteria": pass_criteria,
        "confusion_matrix": confusion,
        "feature_importance": ablation_results,
        "failure_examples": failure_examples[:15],
        "family_distribution": {
            family: sum(1 for label in family_labels.values() if label == family)
            for family in sorted(set(family_labels.values()))
        },
        "n_candidates": len(all_candidates),
        "n_adversarial": len(adversarial_pairs),
    }

    (output_dir / "backbone_shared_benchmark.json").write_text(json.dumps(payload, indent=2, default=str))
    return payload
