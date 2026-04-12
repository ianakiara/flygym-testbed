"""Cross-domain pipeline mock — seam fragility monitor for RAG/agent pipelines.

Implements a mock pipeline: retriever → processor → executor
with perturbation catalog for testing seam fragility in non-FlyGym contexts.

Perturbation types:
1. chunk_mismatch: retriever returns wrong chunks
2. missing_join_fact: processor drops a key fact
3. reordered_outputs: executor receives reordered tool outputs
4. schema_mismatch: field names/types change between stages
5. partial_summary_loss: summarizer drops information
6. stale_memory: memory state is outdated
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class PerturbationType(str, Enum):
    """Types of pipeline perturbations."""

    CHUNK_MISMATCH = "chunk_mismatch"
    MISSING_JOIN_FACT = "missing_join_fact"
    REORDERED_OUTPUTS = "reordered_outputs"
    SCHEMA_MISMATCH = "schema_mismatch"
    PARTIAL_SUMMARY_LOSS = "partial_summary_loss"
    STALE_MEMORY = "stale_memory"


@dataclass(slots=True)
class PipelineStage:
    """A single stage in a pipeline."""

    name: str
    input_schema: list[str]
    output_schema: list[str]

    def process(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Default pass-through processing."""
        return {key: inputs.get(key, None) for key in self.output_schema}


@dataclass(slots=True)
class RetrieverStage(PipelineStage):
    """Retriever: takes query, returns relevant chunks."""

    knowledge_base: list[dict[str, Any]] = field(default_factory=list)

    def process(self, inputs: dict[str, Any]) -> dict[str, Any]:
        query = inputs.get("query", "")
        # Simple keyword matching retrieval
        scored_chunks = []
        for item in self.knowledge_base:
            text = item.get("text", "")
            relevance = sum(1 for word in str(query).split() if word.lower() in text.lower())
            scored_chunks.append((relevance, item))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [c[1] for c in scored_chunks[:3]]
        return {
            "chunks": top_chunks,
            "n_retrieved": len(top_chunks),
            "query": query,
        }


@dataclass(slots=True)
class ProcessorStage(PipelineStage):
    """Processor: takes chunks, extracts structured facts."""

    def process(self, inputs: dict[str, Any]) -> dict[str, Any]:
        chunks = inputs.get("chunks", [])
        facts = []
        for chunk in chunks:
            if isinstance(chunk, dict):
                facts.append({
                    "source": chunk.get("id", "unknown"),
                    "content": chunk.get("text", ""),
                    "confidence": chunk.get("relevance", 0.5),
                })
        return {
            "facts": facts,
            "n_facts": len(facts),
            "synthesis": " ".join(f.get("content", "") for f in facts),
        }


@dataclass(slots=True)
class ExecutorStage(PipelineStage):
    """Executor: takes facts + synthesis, produces final answer."""

    def process(self, inputs: dict[str, Any]) -> dict[str, Any]:
        facts = inputs.get("facts", [])
        synthesis = inputs.get("synthesis", "")
        # Simple answer generation based on fact count and content
        if not facts:
            answer = "No relevant information found."
            confidence = 0.0
        else:
            answer = f"Based on {len(facts)} facts: {synthesis[:200]}"
            confidence = min(1.0, sum(
                f.get("confidence", 0.0) for f in facts
            ) / max(len(facts), 1))
        return {
            "answer": answer,
            "confidence": confidence,
            "n_sources": len(facts),
        }


@dataclass(slots=True)
class Pipeline:
    """A multi-stage pipeline with seam monitoring."""

    stages: list[PipelineStage]

    def run(self, initial_input: dict[str, Any]) -> dict[str, Any]:
        """Execute the full pipeline and return intermediate + final results."""
        current = initial_input
        intermediates: list[dict[str, Any]] = [{"stage": "input", "data": current}]

        for stage in self.stages:
            current = stage.process(current)
            intermediates.append({"stage": stage.name, "data": current})

        return {
            "final_output": current,
            "intermediates": intermediates,
            "n_stages": len(self.stages),
        }


def build_default_pipeline(
    knowledge_base: list[dict[str, Any]] | None = None,
) -> Pipeline:
    """Build the default retriever → processor → executor pipeline."""
    if knowledge_base is None:
        knowledge_base = [
            {"id": "doc1", "text": "The capital of France is Paris.", "relevance": 0.9},
            {"id": "doc2", "text": "Python is a programming language.", "relevance": 0.8},
            {"id": "doc3", "text": "Machine learning uses neural networks.", "relevance": 0.7},
            {"id": "doc4", "text": "The weather is sunny today.", "relevance": 0.3},
            {"id": "doc5", "text": "Seam fragility measures interface quality.", "relevance": 0.6},
        ]

    retriever = RetrieverStage(
        name="retriever",
        input_schema=["query"],
        output_schema=["chunks", "n_retrieved", "query"],
        knowledge_base=knowledge_base,
    )
    processor = ProcessorStage(
        name="processor",
        input_schema=["chunks"],
        output_schema=["facts", "n_facts", "synthesis"],
    )
    executor = ExecutorStage(
        name="executor",
        input_schema=["facts", "synthesis"],
        output_schema=["answer", "confidence", "n_sources"],
    )

    return Pipeline(stages=[retriever, processor, executor])


# ── Perturbation Catalog ─────────────────────────────────────────────────


def apply_perturbation(
    pipeline_result: dict[str, Any],
    perturbation: PerturbationType,
    *,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Apply a perturbation to pipeline intermediates and re-run downstream.

    Returns the perturbed result with metadata about what changed.
    """
    rng = rng or np.random.default_rng(42)
    intermediates = pipeline_result.get("intermediates", [])
    if len(intermediates) < 3:
        return pipeline_result

    perturbed = dict(pipeline_result)
    perturbation_detail: dict[str, Any] = {"type": perturbation.value}

    if perturbation == PerturbationType.CHUNK_MISMATCH:
        # Replace retrieved chunks with irrelevant ones
        retriever_data = intermediates[1]["data"]
        if "chunks" in retriever_data:
            chunks = retriever_data["chunks"]
            if chunks:
                # Swap first chunk with a dummy
                chunks = [{"id": "wrong", "text": "Irrelevant noise.", "relevance": 0.1}] + chunks[1:]
                retriever_data = dict(retriever_data)
                retriever_data["chunks"] = chunks
        perturbation_detail["changed"] = "retriever chunks"

    elif perturbation == PerturbationType.MISSING_JOIN_FACT:
        # Drop a fact from the processor output
        proc_data = intermediates[2]["data"] if len(intermediates) > 2 else {}
        if "facts" in proc_data and proc_data["facts"]:
            proc_data = dict(proc_data)
            proc_data["facts"] = proc_data["facts"][1:]  # drop first fact
            proc_data["n_facts"] = len(proc_data["facts"])
        perturbation_detail["changed"] = "processor facts"

    elif perturbation == PerturbationType.REORDERED_OUTPUTS:
        # Reverse the order of facts
        proc_data = intermediates[2]["data"] if len(intermediates) > 2 else {}
        if "facts" in proc_data and len(proc_data["facts"]) > 1:
            proc_data = dict(proc_data)
            proc_data["facts"] = list(reversed(proc_data["facts"]))
        perturbation_detail["changed"] = "fact ordering"

    elif perturbation == PerturbationType.SCHEMA_MISMATCH:
        # Rename a field
        proc_data = intermediates[2]["data"] if len(intermediates) > 2 else {}
        proc_data = dict(proc_data)
        if "facts" in proc_data:
            proc_data["data_points"] = proc_data.pop("facts")  # rename
        perturbation_detail["changed"] = "schema (facts → data_points)"

    elif perturbation == PerturbationType.PARTIAL_SUMMARY_LOSS:
        # Truncate synthesis
        proc_data = intermediates[2]["data"] if len(intermediates) > 2 else {}
        if "synthesis" in proc_data:
            proc_data = dict(proc_data)
            proc_data["synthesis"] = proc_data["synthesis"][:10]  # severe truncation
        perturbation_detail["changed"] = "synthesis truncation"

    elif perturbation == PerturbationType.STALE_MEMORY:
        # Use outdated knowledge base results
        retriever_data = intermediates[1]["data"]
        if "chunks" in retriever_data:
            retriever_data = dict(retriever_data)
            for chunk in retriever_data["chunks"]:
                if isinstance(chunk, dict):
                    chunk["text"] = chunk.get("text", "") + " [STALE]"
        perturbation_detail["changed"] = "knowledge staleness"

    perturbed["perturbation"] = perturbation_detail
    return perturbed


# ── Seam Metrics for Pipeline ────────────────────────────────────────────


def pipeline_seam_fragility(
    baseline_result: dict[str, Any],
    perturbed_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compute seam fragility metrics for a pipeline across perturbations.

    For each perturbation, compare baseline vs perturbed output quality.
    Seam fragility = how much does perturbation at one stage affect downstream.
    """
    baseline_confidence = baseline_result.get("final_output", {}).get("confidence", 0.0)
    baseline_answer = baseline_result.get("final_output", {}).get("answer", "")

    per_perturbation: list[dict[str, Any]] = []
    for ptype, presult in perturbed_results.items():
        pert_confidence = presult.get("final_output", {}).get("confidence", 0.0)
        pert_answer = presult.get("final_output", {}).get("answer", "")

        confidence_drop = baseline_confidence - pert_confidence
        answer_changed = baseline_answer != pert_answer

        # Local stage quality (did the perturbed stage itself detect the issue?)
        local_quality = pert_confidence  # proxy

        per_perturbation.append({
            "perturbation": ptype,
            "confidence_drop": confidence_drop,
            "answer_changed": answer_changed,
            "local_quality": local_quality,
            "seam_fragility": abs(confidence_drop),
            "structurally_broken": confidence_drop > 0.3,
        })

    mean_fragility = float(np.mean([p["seam_fragility"] for p in per_perturbation])) if per_perturbation else 0.0
    n_broken = sum(1 for p in per_perturbation if p["structurally_broken"])
    looks_fine_but_broken = sum(
        1 for p in per_perturbation
        if p["local_quality"] > 0.3 and p["structurally_broken"]
    )

    return {
        "per_perturbation": per_perturbation,
        "mean_seam_fragility": mean_fragility,
        "max_seam_fragility": float(max(
            (p["seam_fragility"] for p in per_perturbation), default=0.0
        )),
        "n_structurally_broken": n_broken,
        "n_looks_fine_but_broken": looks_fine_but_broken,
        "seam_predicts_better": mean_fragility > 0.1,
    }
