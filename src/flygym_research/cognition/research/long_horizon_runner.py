"""Long-horizon episode runner for PR #5 production validation.

Wraps the existing ``collect_trace_bank`` with configurable episode lengths
(100 / 150 / 200 steps), larger seed sets (10–20), and memory-efficient
batching.  All episodes carry stable, reproducible seeding.
"""

from __future__ import annotations

from ..experiments.exp_sleep_trace_compressor import collect_trace_bank
from ..sleep.trace_schema import TraceEpisode


def collect_long_horizon(
    *,
    max_steps: int = 100,
    n_seeds: int = 10,
    world_modes: list[str] | None = None,
    ablations: list[frozenset[str]] | None = None,
    perturbation_tags: list[str] | None = None,
) -> list[TraceEpisode]:
    """Collect a large trace bank with configurable horizon and seeds.

    Parameters
    ----------
    max_steps : int
        Episode length (100 for standard, 150 for seam/memory, 200 for
        transfer/adversarial).
    n_seeds : int
        Number of seeds (10 for screening, 20 for promoted-result pass).
    world_modes : list[str] | None
        Override default world modes.
    ablations : list[frozenset[str]] | None
        Override default ablation sets.
    perturbation_tags : list[str] | None
        Override default perturbation tags.
    """
    seeds = list(range(n_seeds))
    return collect_trace_bank(
        seeds=seeds,
        max_steps=max_steps,
        world_modes=world_modes,
        ablations=ablations,
        perturbation_tags=perturbation_tags,
    )
