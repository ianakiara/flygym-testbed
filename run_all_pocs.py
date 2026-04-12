"""Run all 15 POC experiments (8 original + 7 new phases) and collect results."""
import json
import traceback
from pathlib import Path

OUTPUT_ROOT = Path("results/poc_validation")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

experiments = [
    # ── Original 8 POCs ──────────────────────────────────────────────────
    ("exp_sleep_trace_compressor", "flygym_research.cognition.experiments.exp_sleep_trace_compressor"),
    ("exp_causal_intervention", "flygym_research.cognition.experiments.exp_causal_intervention"),
    ("exp_drift_cleanup", "flygym_research.cognition.experiments.exp_drift_cleanup"),
    ("exp_interop_alignment", "flygym_research.cognition.experiments.exp_interop_alignment"),
    ("exp_memory_closure", "flygym_research.cognition.experiments.exp_memory_closure"),
    ("exp_seam_repair", "flygym_research.cognition.experiments.exp_seam_repair"),
    ("exp_cross_world_compression", "flygym_research.cognition.experiments.exp_cross_world_compression"),
    ("exp_memory_demand", "flygym_research.cognition.experiments.exp_memory_demand"),
    # ── New Phase POCs ───────────────────────────────────────────────────
    ("exp_hard_memory_benchmark", "flygym_research.cognition.experiments.exp_hard_memory_benchmark"),
    ("exp_functional_transfer", "flygym_research.cognition.experiments.exp_functional_transfer"),
    ("exp_backbone_shared", "flygym_research.cognition.experiments.exp_backbone_shared"),
    ("exp_repair_v2", "flygym_research.cognition.experiments.exp_repair_v2"),
    ("exp_sleep_v2", "flygym_research.cognition.experiments.exp_sleep_v2"),
    ("exp_degenerate_convergence", "flygym_research.cognition.experiments.exp_degenerate_convergence"),
    ("exp_pipeline_seam", "flygym_research.cognition.experiments.exp_pipeline_seam"),
]

all_results = {}
for name, module_path in experiments:
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    try:
        mod = __import__(module_path, fromlist=["run_experiment"])
        out_dir = OUTPUT_ROOT / name
        result = mod.run_experiment(out_dir)
        all_results[name] = {"status": "success", "result": result}
        print("  -> SUCCESS")
        # Print key metrics
        if isinstance(result, dict):
            for k, v in result.items():
                if isinstance(v, (int, float, str, bool)):
                    print(f"     {k}: {v}")
                elif isinstance(v, dict) and len(str(v)) < 500:
                    print(f"     {k}: {json.dumps(v, indent=2, default=str)[:400]}")
    except Exception as e:
        all_results[name] = {"status": "error", "error": str(e), "traceback": traceback.format_exc()}
        print(f"  -> ERROR: {e}")
        traceback.print_exc()

# Save combined results
(OUTPUT_ROOT / "all_poc_results.json").write_text(
    json.dumps(all_results, indent=2, default=str)
)
print(f"\n{'='*60}")
print(f"All results saved to {OUTPUT_ROOT / 'all_poc_results.json'}")
print(f"{'='*60}")

# Summary
print("\n\nFINAL SUMMARY:")
n_pass = 0
n_fail = 0
for name, r in all_results.items():
    status = r['status']
    print(f"  {name}: {status}")
    if status == "success":
        n_pass += 1
    else:
        n_fail += 1
print(f"\n  TOTAL: {n_pass} passed, {n_fail} failed out of {len(experiments)}")
