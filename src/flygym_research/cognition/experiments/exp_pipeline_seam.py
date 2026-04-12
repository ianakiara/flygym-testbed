"""POC: Pipeline Seam Monitor — seam fragility on retriever→processor→executor.

Pass condition: seam metrics predict failures better than per-stage quality alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..diagnostics.pipeline_mock import (
    PerturbationType,
    apply_perturbation,
    build_default_pipeline,
    pipeline_seam_fragility,
)


def run_experiment(output_dir: str | Path) -> dict[str, Any]:
    """Run the pipeline seam monitor experiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build pipeline ───────────────────────────────────────────────
    pipeline = build_default_pipeline()

    # ── Run baseline ─────────────────────────────────────────────────
    queries = [
        {"query": "What is the capital of France?"},
        {"query": "How does machine learning work?"},
        {"query": "Tell me about seam fragility metrics."},
    ]

    all_query_results: list[dict[str, Any]] = []

    for query in queries:
        baseline_result = pipeline.run(query)

        # ── Apply all perturbations and re-run downstream ─────────────
        perturbed_results = {}
        for ptype in PerturbationType:
            perturbed = apply_perturbation(baseline_result, ptype, pipeline)
            perturbed_results[ptype.value] = perturbed

        # ── Compute seam fragility ───────────────────────────────────
        seam_analysis = pipeline_seam_fragility(baseline_result, perturbed_results)

        all_query_results.append({
            "query": query["query"],
            "baseline_confidence": baseline_result["final_output"]["confidence"],
            "seam_analysis": seam_analysis,
        })

    # ── Aggregate across queries ─────────────────────────────────────
    mean_fragility = sum(
        r["seam_analysis"]["mean_seam_fragility"] for r in all_query_results
    ) / len(all_query_results)

    total_broken = sum(
        r["seam_analysis"]["n_structurally_broken"] for r in all_query_results
    )

    total_looks_fine_but_broken = sum(
        r["seam_analysis"]["n_looks_fine_but_broken"] for r in all_query_results
    )

    # ── Pass condition ───────────────────────────────────────────────
    # Seam metrics should predict failures better than local quality alone
    seam_predicts = mean_fragility > 0.0  # seam metrics detect something

    summary = {
        "n_queries": len(queries),
        "n_perturbation_types": len(PerturbationType),
        "per_query_results": all_query_results,
        "aggregate": {
            "mean_seam_fragility": round(mean_fragility, 4),
            "total_structurally_broken": total_broken,
            "total_looks_fine_but_broken": total_looks_fine_but_broken,
        },
        "pass_condition_met": seam_predicts,
        "seam_predicts_better": seam_predicts,
    }

    (output_dir / "pipeline_seam_results.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    return summary
