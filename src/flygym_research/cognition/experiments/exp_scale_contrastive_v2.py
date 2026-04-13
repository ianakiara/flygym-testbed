"""EXP 6 — Scale Law Contrastive Benchmark (PR #5 priority 7).

Moves from "stable features survive" to actual real vs fake separation.
Constructs explicitly scale-sensitive fake metrics and tests whether
scale transforms can discriminate them from real structural features.

Pass criteria:
  - real features stable (CV < threshold)
  - fake features unstable/collapsing
  - strong separation between families

Includes baseline (no transform), method (real features), ablation (fake features).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..research.long_horizon_runner import collect_long_horizon
from ..research.scale_transforms import (
    SCALE_TRANSFORMS,
    apply_scale_transform,
    compute_fake_metrics,
    compute_real_metrics,
)



# ---------------------------------------------------------------------------
# Stability computation
# ---------------------------------------------------------------------------

def _stability_coefficient(values: list[float]) -> float:
    """CV with special handling for zero/constant sequences."""
    arr = np.array(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if len(finite) < 2:
        return 0.0
    mean = np.mean(finite)
    std = np.std(finite)
    if abs(mean) < 1e-12:
        # Zero mean: if std is also ~0, it's stable (constant zero)
        return 0.0 if std < 1e-10 else float("inf")
    return float(std / abs(mean))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    output_dir: str | Path = "results/exp_scale_contrastive_v2",
    *,
    episode_steps: int = 100,
    n_seeds: int = 10,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect episodes
    episodes = collect_long_horizon(max_steps=episode_steps, n_seeds=n_seeds)

    if not episodes:
        payload = {"error": "No episodes"}
        (output_dir / "scale_contrastive_v2.json").write_text(json.dumps(payload, indent=2))
        return payload

    transform_names = list(SCALE_TRANSFORMS.keys())

    # For each episode, compute real and fake metrics at each scale
    real_property_names = list(compute_real_metrics([]).keys()) if True else []
    fake_property_names = list(compute_fake_metrics([]).keys()) if True else []

    # Collect per-property, per-transform values across all episodes
    real_values: dict[str, dict[str, list[float]]] = {
        prop: {tx: [] for tx in transform_names} for prop in real_property_names
    }
    fake_values: dict[str, dict[str, list[float]]] = {
        prop: {tx: [] for tx in transform_names} for prop in fake_property_names
    }

    for ep in episodes:
        for tx_name in transform_names:
            transformed = apply_scale_transform(ep.transitions, tx_name)
            if not transformed:
                continue

            real_m = compute_real_metrics(transformed)
            fake_m = compute_fake_metrics(transformed)

            for prop in real_property_names:
                real_values[prop][tx_name].append(real_m.get(prop, 0.0))
            for prop in fake_property_names:
                fake_values[prop][tx_name].append(fake_m.get(prop, 0.0))

    # Compute stability (CV across transforms) for each property
    real_stability: dict[str, dict] = {}
    for prop in real_property_names:
        # For each episode, get the mean value at each transform
        per_transform_means = {
            tx: float(np.mean(vals)) if vals else 0.0
            for tx, vals in real_values[prop].items()
        }
        cv = _stability_coefficient(list(per_transform_means.values()))
        real_stability[prop] = {
            "cv": cv,
            "is_stable": cv < 0.5 and np.isfinite(cv),
            "per_transform_means": per_transform_means,
        }

    fake_stability: dict[str, dict] = {}
    for prop in fake_property_names:
        per_transform_means = {
            tx: float(np.mean(vals)) if vals else 0.0
            for tx, vals in fake_values[prop].items()
        }
        cv = _stability_coefficient(list(per_transform_means.values()))
        fake_stability[prop] = {
            "cv": cv,
            "is_stable": cv < 0.5 and np.isfinite(cv),
            "per_transform_means": per_transform_means,
        }

    # Separation metrics
    real_cvs = [s["cv"] for s in real_stability.values() if np.isfinite(s["cv"])]
    fake_cvs = [s["cv"] for s in fake_stability.values() if np.isfinite(s["cv"])]

    mean_real_cv = float(np.mean(real_cvs)) if real_cvs else 0.0
    mean_fake_cv = float(np.mean(fake_cvs)) if fake_cvs else 0.0

    n_real_stable = sum(1 for s in real_stability.values() if s["is_stable"])
    n_fake_unstable = sum(1 for s in fake_stability.values() if not s["is_stable"])

    # Rank stability: do real properties maintain their relative ordering?
    rank_stability_real = {}
    for prop in real_property_names:
        per_tx = real_values[prop]
        # For each pair of transforms, compute rank correlation of episode values
        rank_corrs = []
        txs = [tx for tx in transform_names if len(per_tx.get(tx, [])) > 3]
        for i in range(len(txs)):
            for j in range(i + 1, len(txs)):
                vals_i = np.array(per_tx[txs[i]])
                vals_j = np.array(per_tx[txs[j]])
                n = min(len(vals_i), len(vals_j))
                if n < 3:
                    continue
                # Spearman-like: correlation of ranks
                if np.std(vals_i[:n]) > 1e-10 and np.std(vals_j[:n]) > 1e-10:
                    corr = float(np.corrcoef(vals_i[:n], vals_j[:n])[0, 1])
                    if np.isfinite(corr):
                        rank_corrs.append(corr)
        rank_stability_real[prop] = float(np.mean(rank_corrs)) if rank_corrs else 0.0

    # Cross-transform agreement for real vs fake
    real_agreement = float(np.mean(list(rank_stability_real.values()))) if rank_stability_real else 0.0

    # Collapse rate: fraction of fake metrics that change > 50% between original and any transform
    collapse_count = 0
    collapse_total = 0
    for prop in fake_property_names:
        orig_vals = fake_values[prop].get("original", [])
        if not orig_vals:
            continue
        orig_mean = float(np.mean(orig_vals))
        for tx in transform_names:
            if tx == "original":
                continue
            tx_vals = fake_values[prop].get(tx, [])
            if not tx_vals:
                continue
            tx_mean = float(np.mean(tx_vals))
            collapse_total += 1
            if abs(orig_mean) > 1e-8 and abs(tx_mean - orig_mean) / abs(orig_mean) > 0.5:
                collapse_count += 1

    collapse_rate = float(collapse_count / max(collapse_total, 1))

    # Classifier accuracy: can we distinguish real from fake by CV?
    all_labels = []
    all_cvs_for_classifier = []
    for prop in real_property_names:
        cv = real_stability[prop]["cv"]
        if np.isfinite(cv):
            all_labels.append(True)  # real
            all_cvs_for_classifier.append(cv)
    for prop in fake_property_names:
        cv = fake_stability[prop]["cv"]
        if np.isfinite(cv):
            all_labels.append(False)  # fake
            all_cvs_for_classifier.append(cv)

    # Simple threshold classifier: real = low CV, fake = high CV
    if all_cvs_for_classifier:
        threshold = float(np.median(all_cvs_for_classifier))
        predictions = [cv < threshold for cv in all_cvs_for_classifier]
        accuracy = float(np.mean([
            1.0 if pred == label else 0.0
            for pred, label in zip(predictions, all_labels)
        ]))
    else:
        accuracy = 0.0
        threshold = 0.0

    payload = {
        "real_stability": {k: {kk: vv for kk, vv in v.items() if kk != "per_transform_means"}
                          for k, v in real_stability.items()},
        "fake_stability": {k: {kk: vv for kk, vv in v.items() if kk != "per_transform_means"}
                          for k, v in fake_stability.items()},
        "pass_criteria": {
            "real_features_stable": n_real_stable >= len(real_property_names) * 0.6,
            "n_real_stable": n_real_stable,
            "n_real_total": len(real_property_names),
            "fake_features_unstable": n_fake_unstable >= len(fake_property_names) * 0.5,
            "n_fake_unstable": n_fake_unstable,
            "n_fake_total": len(fake_property_names),
            "strong_separation": mean_fake_cv > mean_real_cv * 1.5,
            "mean_real_cv": mean_real_cv,
            "mean_fake_cv": mean_fake_cv,
        },
        "separation_metrics": {
            "classifier_accuracy": accuracy,
            "classifier_threshold": threshold,
            "collapse_rate_fake": collapse_rate,
            "rank_stability_real": rank_stability_real,
            "cross_transform_agreement_real": real_agreement,
        },
        "config": {
            "n_episodes": len(episodes),
            "n_transforms": len(transform_names),
            "n_real_properties": len(real_property_names),
            "n_fake_properties": len(fake_property_names),
            "episode_steps": episode_steps,
            "n_seeds": n_seeds,
        },
    }

    (output_dir / "scale_contrastive_v2.json").write_text(
        json.dumps(payload, indent=2, default=str),
    )
    return payload
