"""EXP 7 — Observer-Family Benchmark v2 (PR #5 priority 5).

Reconfirms the translation advantage under 9 genuinely different observer
distortion families with stronger perturbations.

Pass criteria:
  - translated > raw across most families
  - large effect size remains
  - raw locality no longer looks competitive
  - nonlinear/hard transforms don't collapse translation

Includes baseline (raw), method (translated), ablation (per-family breakdown).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..metrics.interoperability_metrics import (
    raw_latent_alignment,
    translated_latent_alignment,
)
from ..research.long_horizon_runner import collect_long_horizon
from ..research.observer_families import OBSERVER_FAMILIES, apply_observer_perturbation


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    output_dir: str | Path = "results/exp_observer_interop_v2",
    *,
    episode_steps: int = 100,
    n_seeds: int = 10,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # Collect episodes
    episodes = collect_long_horizon(max_steps=episode_steps, n_seeds=n_seeds)

    if len(episodes) < 2:
        payload = {"error": "Not enough episodes"}
        (output_dir / "observer_interop_v2.json").write_text(json.dumps(payload, indent=2))
        return payload

    # Select episode pairs for comparison
    pairs = []
    for i in range(0, min(len(episodes) - 1, 40), 2):
        pairs.append((episodes[i], episodes[i + 1]))

    family_names = list(OBSERVER_FAMILIES.keys())

    # Per-family results
    per_family: dict[str, dict] = {}
    all_raw_corrs = []
    all_trans_corrs = []

    for family_name in family_names:
        family_raw_mse = []
        family_trans_mse = []
        family_raw_corr = []
        family_trans_corr = []
        family_improvement = []

        for ep_a, ep_b in pairs:
            # Original transitions
            trans_a = ep_a.transitions
            trans_b = ep_b.transitions

            # Apply perturbation to trans_b
            perturbed_b = apply_observer_perturbation(
                trans_b, family_name, rng=rng,
            )

            # Raw alignment: original A vs perturbed B
            raw = raw_latent_alignment(trans_a, perturbed_b)
            raw_corr = float(raw.get("raw_alignment", 0.0))

            # Translated alignment: fit linear map A → perturbed B
            translated = translated_latent_alignment(trans_a, perturbed_b)
            trans_corr = float(translated.get("translated_alignment", 0.0))
            trans_residual = float(translated.get("translation_residual_norm", 0.0))

            # Compute MSE proxy from residual norm
            raw_mse = 1.0 - abs(raw_corr)  # proxy: lower correlation = higher mse
            trans_mse = trans_residual

            improvement = trans_corr - raw_corr

            family_raw_mse.append(raw_mse)
            family_trans_mse.append(trans_mse)
            family_raw_corr.append(raw_corr)
            family_trans_corr.append(trans_corr)
            family_improvement.append(improvement)
            all_raw_corrs.append(raw_corr)
            all_trans_corrs.append(trans_corr)

        per_family[family_name] = {
            "n_pairs": len(pairs),
            "raw_mse_mean": float(np.mean(family_raw_mse)) if family_raw_mse else 0.0,
            "translated_mse_mean": float(np.mean(family_trans_mse)) if family_trans_mse else 0.0,
            "raw_corr_mean": float(np.mean(family_raw_corr)) if family_raw_corr else 0.0,
            "translated_corr_mean": float(np.mean(family_trans_corr)) if family_trans_corr else 0.0,
            "improvement_mean": float(np.mean(family_improvement)) if family_improvement else 0.0,
            "improvement_std": float(np.std(family_improvement)) if family_improvement else 0.0,
            "translated_gt_raw": float(np.mean(family_trans_corr)) > float(np.mean(family_raw_corr)),
            # Effect size (Cohen's d)
            "effect_size": float(
                np.mean(family_improvement) / (np.std(family_improvement) + 1e-8)
            ) if family_improvement else 0.0,
        }

    # Overall metrics
    overall_raw_corr = float(np.mean(all_raw_corrs)) if all_raw_corrs else 0.0
    overall_trans_corr = float(np.mean(all_trans_corrs)) if all_trans_corrs else 0.0

    # Count how many families show translation advantage
    families_with_advantage = sum(
        1 for f in per_family.values() if f["translated_gt_raw"]
    )
    families_with_large_effect = sum(
        1 for f in per_family.values() if f["effect_size"] > 0.5
    )

    # Hard transforms: check nonlinear and mixed specifically
    hard_families = ["nonlinear", "mixed", "permutation", "sign_flip"]
    hard_advantage = sum(
        1 for fn in hard_families
        if fn in per_family and per_family[fn]["translated_gt_raw"]
    )

    # Translation robustness: std of improvement across families
    all_improvements = [f["improvement_mean"] for f in per_family.values()]
    robustness_cv = float(
        np.std(all_improvements) / (abs(np.mean(all_improvements)) + 1e-8)
    ) if all_improvements else 0.0

    payload = {
        "per_family": per_family,
        "pass_criteria": {
            "translated_gt_raw_most_families": families_with_advantage >= len(family_names) * 0.6,
            "n_families_with_advantage": families_with_advantage,
            "n_families_total": len(family_names),
            "large_effect_size_remains": families_with_large_effect >= 3,
            "n_families_large_effect": families_with_large_effect,
            "raw_not_competitive": overall_raw_corr < overall_trans_corr,
            "hard_transforms_survive": hard_advantage >= 2,
            "n_hard_families_surviving": hard_advantage,
        },
        "overall": {
            "raw_corr_mean": overall_raw_corr,
            "translated_corr_mean": overall_trans_corr,
            "improvement": overall_trans_corr - overall_raw_corr,
            "robustness_cv": robustness_cv,
        },
        "config": {
            "n_observer_families": len(family_names),
            "n_pairs": len(pairs),
            "episode_steps": episode_steps,
            "n_seeds": n_seeds,
            "n_episodes": len(episodes),
        },
    }

    (output_dir / "observer_interop_v2.json").write_text(
        json.dumps(payload, indent=2, default=str),
    )
    return payload
