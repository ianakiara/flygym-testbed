"""Diagnostics package — Task Complexity Meter, Metric Auditor,
Degenerate Convergence Detector, and Pipeline Mock."""

from .degenerate_convergence import (
    ConvergenceAnalysis,
    ConvergenceType,
    compute_diversity,
    compute_recovery,
    compute_transfer,
    construct_degenerate_scenario,
    detect_convergence,
)
from .metric_auditor import (
    AuditResult,
    MetricAuditReport,
    audit_metric,
    check_averaging_distortion,
    check_constant_output,
    check_permutation_invariance,
    check_scale_invariance,
)
from .pipeline_mock import (
    ExecutorStage,
    Pipeline,
    PipelineStage,
    PerturbationType,
    ProcessorStage,
    RetrieverStage,
    apply_perturbation,
    build_default_pipeline,
    pipeline_seam_fragility,
)
from .task_complexity_meter import (
    TaskComplexityResult,
    batch_complexity_assessment,
    classify_complexity,
    compute_task_complexity,
)

__all__ = [
    "AuditResult",
    "ConvergenceAnalysis",
    "ConvergenceType",
    "ExecutorStage",
    "MetricAuditReport",
    "PerturbationType",
    "Pipeline",
    "PipelineStage",
    "ProcessorStage",
    "RetrieverStage",
    "TaskComplexityResult",
    "apply_perturbation",
    "audit_metric",
    "batch_complexity_assessment",
    "build_default_pipeline",
    "check_averaging_distortion",
    "check_constant_output",
    "check_permutation_invariance",
    "check_scale_invariance",
    "classify_complexity",
    "compute_diversity",
    "compute_recovery",
    "compute_task_complexity",
    "compute_transfer",
    "construct_degenerate_scenario",
    "detect_convergence",
    "pipeline_seam_fragility",
]
