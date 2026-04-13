#!/usr/bin/env python3
"""PR #5 production experiment runner.

Runs all 7 v2 experiments with production-grade parameters:
- 100-200 step episodes
- 200-300+ candidates
- 10-20 seeds
- 25% adversarial cases
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

RESULTS_DIR = Path("/home/ubuntu/repos/flygym-testbed/results_pr5")
RESULTS_DIR.mkdir(exist_ok=True)


def run_exp3_deep_memory():
    """EXP 3: Deep memory benchmark v2 (Priority 1)."""
    print("=" * 60)
    print("EXP 3: Deep Memory Benchmark v2")
    print("  Episodes: 150 steps | Seeds: 10")
    print("=" * 60)
    from flygym_research.cognition.experiments.exp_deep_memory_v2 import run_experiment
    out = RESULTS_DIR / "exp3_deep_memory_v2"
    out.mkdir(exist_ok=True)
    t0 = time.time()
    results = run_experiment(output_dir=out, episode_steps=150, n_seeds=10)
    elapsed = time.time() - t0
    results["_runtime_seconds"] = elapsed
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Pass criteria: {json.dumps(results['pass_criteria'], indent=2)}")
    return results


def run_exp1_long_composition():
    """EXP 1: Long-horizon composition stress (Priority 2)."""
    print("=" * 60)
    print("EXP 1: Long-horizon Composition Stress")
    print("  Episodes: 150 steps | Seeds: 10")
    print("=" * 60)
    from flygym_research.cognition.experiments.exp_long_composition import run_experiment
    out = RESULTS_DIR / "exp1_long_composition"
    out.mkdir(exist_ok=True)
    t0 = time.time()
    results = run_experiment(output_dir=out, episode_steps=150, n_seeds=10)
    elapsed = time.time() - t0
    results["_runtime_seconds"] = elapsed
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Pass criteria: {json.dumps(results['pass_criteria'], indent=2)}")
    return results


def run_exp2_backbone_shared():
    """EXP 2: BackboneShared adversarial v2 (Priority 3)."""
    print("=" * 60)
    print("EXP 2: BackboneShared Adversarial Benchmark v2")
    print("  Episodes: 150 steps | Seeds: 10 | Min pool: 200")
    print("=" * 60)
    from flygym_research.cognition.experiments.exp_backbone_shared_v2 import run_experiment
    out = RESULTS_DIR / "exp2_backbone_shared_v2"
    out.mkdir(exist_ok=True)
    t0 = time.time()
    results = run_experiment(output_dir=out, episode_steps=150, n_seeds=10, min_pool_size=200)
    elapsed = time.time() - t0
    results["_runtime_seconds"] = elapsed
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Pass criteria: {json.dumps(results['pass_criteria'], indent=2)}")
    return results


def run_exp5_selector_stress():
    """EXP 5: Survival vs scalar selector stress (Priority 4)."""
    print("=" * 60)
    print("EXP 5: Selector Stress Test v2")
    print("  Episodes: 150 steps | Seeds: 10 | Min pool: 250")
    print("=" * 60)
    from flygym_research.cognition.experiments.exp_selector_stress_v2 import run_experiment
    out = RESULTS_DIR / "exp5_selector_stress_v2"
    out.mkdir(exist_ok=True)
    t0 = time.time()
    results = run_experiment(output_dir=out, episode_steps=150, n_seeds=10, min_pool_size=250)
    elapsed = time.time() - t0
    results["_runtime_seconds"] = elapsed
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Pass criteria: {json.dumps(results['pass_criteria'], indent=2)}")
    return results


def run_exp7_observer_interop():
    """EXP 7: Observer-family v2 (Priority 5)."""
    print("=" * 60)
    print("EXP 7: Observer-Family Benchmark v2")
    print("  Episodes: 100 steps | Seeds: 10 | 9 observer families")
    print("=" * 60)
    from flygym_research.cognition.experiments.exp_observer_interop_v2 import run_experiment
    out = RESULTS_DIR / "exp7_observer_interop_v2"
    out.mkdir(exist_ok=True)
    t0 = time.time()
    results = run_experiment(output_dir=out, episode_steps=100, n_seeds=10)
    elapsed = time.time() - t0
    results["_runtime_seconds"] = elapsed
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Pass criteria: {json.dumps(results['pass_criteria'], indent=2)}")
    return results


def run_exp4_portable_replay():
    """EXP 4: Portable replay v2 (Priority 6)."""
    print("=" * 60)
    print("EXP 4: Portable Replay Benchmark v2")
    print("  Episodes: 200 steps | Seeds: 10 | 5-world replay")
    print("=" * 60)
    from flygym_research.cognition.experiments.exp_portable_replay_v2 import run_experiment
    out = RESULTS_DIR / "exp4_portable_replay_v2"
    out.mkdir(exist_ok=True)
    t0 = time.time()
    results = run_experiment(output_dir=out, episode_steps=200, n_seeds=10)
    elapsed = time.time() - t0
    results["_runtime_seconds"] = elapsed
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Pass criteria: {json.dumps(results['pass_criteria'], indent=2)}")
    return results


def run_exp6_scale_contrastive():
    """EXP 6: Scale law contrastive (Priority 7)."""
    print("=" * 60)
    print("EXP 6: Scale Law Contrastive Benchmark")
    print("  Episodes: 100 steps | Seeds: 10 | 7+ transforms")
    print("=" * 60)
    from flygym_research.cognition.experiments.exp_scale_contrastive_v2 import run_experiment
    out = RESULTS_DIR / "exp6_scale_contrastive_v2"
    out.mkdir(exist_ok=True)
    t0 = time.time()
    results = run_experiment(output_dir=out, episode_steps=100, n_seeds=10)
    elapsed = time.time() - t0
    results["_runtime_seconds"] = elapsed
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Pass criteria: {json.dumps(results['pass_criteria'], indent=2)}")
    return results


def classify_branches(all_results):
    """Classify branches as promoted / active / retired."""
    verdicts = {}

    # EXP 3: Deep Memory
    exp3 = all_results.get("exp3", {}).get("pass_criteria", {})
    memory_wins = exp3.get("memory_gt_reactive_4_of_6_tasks", False)
    causal_depth = exp3.get("causal_depth_gt_1", False)
    attention_wins = exp3.get("attention_gt_scalar_on_selective", False)
    if memory_wins and causal_depth and attention_wins:
        verdicts["deep_memory"] = "PROMOTED"
    elif memory_wins:
        verdicts["deep_memory"] = "ACTIVE (memory advantage confirmed, selective memory unproven)"
    else:
        verdicts["deep_memory"] = "RETIRED"

    # EXP 1: Composition Law
    exp1 = all_results.get("exp1", {}).get("pass_criteria", {})
    boundary_wins = exp1.get("boundary_gt_bulk_win_rate", 0) >= 0.85
    corner_wins = exp1.get("corner_gt_boundary_win_rate", 0) >= 0.80
    seam_rho = exp1.get("seam_rho", 0) > 0.7
    seam_var = exp1.get("seam_variance", 0) > 0
    if boundary_wins and corner_wins and seam_rho:
        verdicts["composition_law"] = "PROMOTED"
    elif boundary_wins and corner_wins:
        verdicts["composition_law"] = "ACTIVE (ordering confirmed, seam metric weak)"
        verdicts["seam_scalar_metric"] = "RETIRED" if not seam_var else "ACTIVE"
    else:
        verdicts["composition_law"] = "RETIRED"

    # EXP 2: BackboneShared
    exp2 = all_results.get("exp2", {}).get("pass_criteria", {})
    auc = exp2.get("backbone_shared_auc", 0)
    beats_omega = exp2.get("beats_omega_by_008", False)
    if auc >= 0.90 and beats_omega:
        verdicts["backbone_shared"] = "PROMOTED"
    elif auc >= 0.80:
        verdicts["backbone_shared"] = "ACTIVE (AUC >= 0.80 but not strong)"
    elif auc >= 0.70:
        verdicts["backbone_shared"] = "ACTIVE (marginal, needs work)"
    else:
        verdicts["backbone_shared"] = "RETIRED in FlyGym (AUC < 0.70)"

    # EXP 5: Selector
    exp5 = all_results.get("exp5", {}).get("pass_criteria", {})
    survival_wins = exp5.get("survival_lower_catastrophic_than_scalar", False)
    scalar_dangerous = exp5.get("scalar_admits_more_dangerous", False)
    if survival_wins and scalar_dangerous:
        verdicts["survival_selector"] = "PROMOTED"
    elif survival_wins:
        verdicts["survival_selector"] = "ACTIVE (safety filter only)"
    else:
        verdicts["survival_selector"] = "RETIRED as general selector"

    # EXP 7: Observer Interop
    exp7 = all_results.get("exp7", {}).get("pass_criteria", {})
    translated_wins = exp7.get("translated_gt_raw_most_families", False)
    hard_survives = exp7.get("hard_transforms_survive", False)
    if translated_wins and hard_survives:
        verdicts["observer_interop"] = "PROMOTED"
    elif translated_wins:
        verdicts["observer_interop"] = "ACTIVE (works for easy transforms only)"
    else:
        verdicts["observer_interop"] = "RETIRED"

    # EXP 4: Portable Replay
    exp4 = all_results.get("exp4", {}).get("pass_criteria", {})
    portable_wins = exp4.get("portable_gt_local_on_transfer", False)
    fp_rate = exp4.get("false_portable_detection_rate", 0.0)
    false_portable_precise = fp_rate >= 0.5  # 50%+ detection = "high"
    if portable_wins and false_portable_precise:
        verdicts["tier_taxonomy"] = "PROMOTED"
    elif portable_wins:
        verdicts["tier_taxonomy"] = "ACTIVE (tiers separate but noisy)"
    else:
        verdicts["tier_taxonomy"] = "RETIRED (use direct transfer scoring)"

    # EXP 6: Scale Law
    exp6 = all_results.get("exp6", {}).get("pass_criteria", {})
    real_stable = exp6.get("real_features_stable", False)
    fake_unstable = exp6.get("fake_features_unstable", False)
    strong_sep = exp6.get("strong_separation", False)
    if real_stable and fake_unstable and strong_sep:
        verdicts["scale_law"] = "PROMOTED"
    elif real_stable and strong_sep:
        verdicts["scale_law"] = "ACTIVE (real stable, strong separation, fake don't individually collapse)"
    elif real_stable and fake_unstable:
        verdicts["scale_law"] = "ACTIVE (separation exists but weak)"
    else:
        verdicts["scale_law"] = "RETIRED (no real/fake discrimination)"

    return verdicts


def main():
    all_results = {}
    total_t0 = time.time()

    # Priority 1: Deep Memory
    try:
        all_results["exp3"] = run_exp3_deep_memory()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_results["exp3"] = {"error": str(e)}

    # Priority 2: Long-horizon Composition
    try:
        all_results["exp1"] = run_exp1_long_composition()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_results["exp1"] = {"error": str(e)}

    # Priority 3: BackboneShared
    try:
        all_results["exp2"] = run_exp2_backbone_shared()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_results["exp2"] = {"error": str(e)}

    # Priority 4: Selector Stress
    try:
        all_results["exp5"] = run_exp5_selector_stress()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_results["exp5"] = {"error": str(e)}

    # Priority 5: Observer Interop
    try:
        all_results["exp7"] = run_exp7_observer_interop()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_results["exp7"] = {"error": str(e)}

    # Priority 6: Portable Replay
    try:
        all_results["exp4"] = run_exp4_portable_replay()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_results["exp4"] = {"error": str(e)}

    # Priority 7: Scale Contrastive
    try:
        all_results["exp6"] = run_exp6_scale_contrastive()
    except Exception as e:
        print(f"  FAILED: {e}")
        all_results["exp6"] = {"error": str(e)}

    total_elapsed = time.time() - total_t0

    # Classify branches
    verdicts = classify_branches(all_results)

    # Summary
    print("\n" + "=" * 60)
    print("PR #5 PRODUCTION VALIDATION COMPLETE")
    print(f"Total runtime: {total_elapsed:.1f}s")
    print("=" * 60)

    print("\n--- BRANCH VERDICTS ---")
    for branch, verdict in verdicts.items():
        print(f"  {branch}: {verdict}")

    print("\n--- PASS/FAIL CRITERIA ---")
    for exp_key, exp_data in all_results.items():
        pc = exp_data.get("pass_criteria", {})
        if pc:
            print(f"\n  {exp_key}:")
            for k, v in pc.items():
                print(f"    {k}: {v}")

    # Write combined results
    combined = {
        "verdicts": verdicts,
        "total_runtime_seconds": total_elapsed,
        "experiments": {},
    }
    for exp_key, exp_data in all_results.items():
        combined["experiments"][exp_key] = {
            "pass_criteria": exp_data.get("pass_criteria", {}),
            "config": exp_data.get("config", {}),
            "runtime": exp_data.get("_runtime_seconds", 0),
        }

    with open(RESULTS_DIR / "combined_results.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)

    print(f"\nResults written to {RESULTS_DIR}")
    return combined


if __name__ == "__main__":
    result = main()
    sys.exit(0)
