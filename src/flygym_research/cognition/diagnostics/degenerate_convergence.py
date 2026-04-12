"""Degenerate convergence detector — separates healthy shared structure from collapse.

Key discriminator: recovery + transfer.
- Healthy: high Ω (BackboneShared) + high recovery after perturbation + good transfer
- Degenerate: high Ω + low recovery + poor transfer

Degenerate scenarios:
1. Reward collapse: reward function collapses diversity
2. Stress convergence: perturbation pushes all agents into same basin
3. Compression narrowing: sleep compression removes too much diversity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from ..interfaces import StepTransition
from ..metrics import seam_fragility
from ..metrics.interoperability_metrics import extract_state_matrix


class ConvergenceType(str, Enum):
    """Classification of convergence pattern."""

    HEALTHY = "healthy"
    DEGENERATE = "degenerate"
    AMBIGUOUS = "ambiguous"


@dataclass(slots=True)
class ConvergenceAnalysis:
    """Full analysis of convergence health."""

    convergence_type: ConvergenceType
    omega_score: float
    recovery_score: float
    transfer_score: float
    diversity_score: float
    detail: dict[str, float] = field(default_factory=dict)


def compute_diversity(
    transitions_dict: dict[str, list[StepTransition]],
) -> float:
    """Compute behavioral diversity across controllers.

    Uses coefficient of variation of reward trajectories as proxy.
    High CV → diverse behaviors. Low CV → convergent behaviors.
    """
    if len(transitions_dict) < 2:
        return 0.0

    min_len = min(len(t) for t in transitions_dict.values())
    if min_len < 2:
        return 0.0

    names = sorted(transitions_dict.keys())
    rewards = np.array([
        [t.reward for t in transitions_dict[n][:min_len]]
        for n in names
    ])

    # Per-timestep variance across controllers
    per_step_var = np.var(rewards, axis=0)
    mean_var = float(np.mean(per_step_var))

    # Overall reward scale
    overall_mean = float(np.mean(np.abs(rewards)))
    if overall_mean < 1e-10:
        return 0.0

    # CV-like diversity score
    return float(np.sqrt(mean_var) / overall_mean)


def compute_recovery(
    baseline_transitions: dict[str, list[StepTransition]],
    perturbed_transitions: dict[str, list[StepTransition]],
) -> float:
    """Compute recovery: how well do controllers recover from perturbation?

    Recovery = correlation between baseline and perturbed reward trajectories.
    High correlation → controllers recovered their behavior.
    Low correlation → perturbation permanently changed behavior.
    """
    common = sorted(set(baseline_transitions.keys()) & set(perturbed_transitions.keys()))
    if len(common) < 1:
        return 0.0

    recovery_scores = []
    for name in common:
        base = baseline_transitions[name]
        pert = perturbed_transitions[name]
        n = min(len(base), len(pert))
        if n < 4:
            continue
        r_base = np.array([t.reward for t in base[:n]])
        r_pert = np.array([t.reward for t in pert[:n]])

        if np.std(r_base) < 1e-10 or np.std(r_pert) < 1e-10:
            recovery_scores.append(0.0)
            continue

        corr = float(np.corrcoef(r_base, r_pert)[0, 1])
        if np.isnan(corr):
            corr = 0.0
        recovery_scores.append(corr)

    return float(np.mean(recovery_scores)) if recovery_scores else 0.0


def compute_transfer(
    source_transitions: dict[str, list[StepTransition]],
    target_transitions: dict[str, list[StepTransition]],
) -> float:
    """Compute transfer quality: how well does structure transfer to new world?

    Uses state-space translation R² as proxy for transferability.
    """
    common = sorted(set(source_transitions.keys()) & set(target_transitions.keys()))
    if len(common) < 1:
        return 0.0

    r2_scores = []
    for name in common:
        src = source_transitions[name]
        tgt = target_transitions[name]
        n = min(len(src), len(tgt))
        if n < 4:
            continue

        Za = extract_state_matrix(src[:n])
        Zb = extract_state_matrix(tgt[:n])

        # Filter constant columns
        active_a = np.std(Za, axis=0) > 1e-10
        active_b = np.std(Zb, axis=0) > 1e-10
        Za_f = Za[:, active_a]
        Zb_f = Zb[:, active_b]

        if Za_f.shape[1] < 1 or Zb_f.shape[1] < 1:
            r2_scores.append(0.0)
            continue

        # Standardize
        Za_n = (Za_f - Za_f.mean(0)) / np.maximum(Za_f.std(0), 1e-10)
        Zb_n = (Zb_f - Zb_f.mean(0)) / np.maximum(Zb_f.std(0), 1e-10)

        src_aug = np.hstack([Za_n, np.ones((Za_n.shape[0], 1))])
        try:
            T, _, _, _ = np.linalg.lstsq(src_aug, Zb_n, rcond=None)
            pred = src_aug @ T
            ss_res = float(np.sum((Zb_n - pred) ** 2))
            ss_tot = float(np.sum((Zb_n - Zb_n.mean(0)) ** 2))
            r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
        except np.linalg.LinAlgError:
            r2 = 0.0
        r2_scores.append(r2)

    return float(np.mean(r2_scores)) if r2_scores else 0.0


def detect_convergence(
    baseline_transitions: dict[str, list[StepTransition]],
    perturbed_transitions: dict[str, list[StepTransition]] | None = None,
    transfer_transitions: dict[str, list[StepTransition]] | None = None,
    *,
    omega_score: float | None = None,
    recovery_threshold: float = 0.5,
    transfer_threshold: float = 0.3,
    diversity_threshold: float = 0.1,
) -> ConvergenceAnalysis:
    """Detect whether convergence is healthy or degenerate.

    Parameters
    ----------
    baseline_transitions : dict
        Controller name → transitions in the baseline condition.
    perturbed_transitions : dict, optional
        Same controllers after perturbation (for recovery test).
    transfer_transitions : dict, optional
        Same controllers in a different world (for transfer test).
    omega_score : float, optional
        Pre-computed BackboneShared score. If None, estimated from data.
    """
    # Diversity
    diversity = compute_diversity(baseline_transitions)

    # Recovery
    if perturbed_transitions:
        recovery = compute_recovery(baseline_transitions, perturbed_transitions)
    else:
        recovery = 0.5  # neutral if no perturbation data

    # Transfer
    if transfer_transitions:
        transfer = compute_transfer(baseline_transitions, transfer_transitions)
    else:
        transfer = 0.5  # neutral if no transfer data

    # Omega (BackboneShared proxy)
    if omega_score is None:
        # Estimate from seam fragility + diversity
        seam_scores = []
        for name, trans in baseline_transitions.items():
            sf = seam_fragility(trans)["seam_fragility"]
            seam_scores.append(sf)
        mean_seam = float(np.mean(seam_scores)) if seam_scores else 0.0
        omega_score = max(0.0, 1.0 - mean_seam)  # rough proxy

    # Classification
    high_omega = omega_score > 0.3
    high_recovery = recovery > recovery_threshold
    good_transfer = transfer > transfer_threshold
    good_diversity = diversity > diversity_threshold

    if high_omega and high_recovery and (good_transfer or good_diversity):
        ctype = ConvergenceType.HEALTHY
    elif high_omega and (not high_recovery or not good_transfer):
        ctype = ConvergenceType.DEGENERATE
    else:
        ctype = ConvergenceType.AMBIGUOUS

    return ConvergenceAnalysis(
        convergence_type=ctype,
        omega_score=omega_score,
        recovery_score=recovery,
        transfer_score=transfer,
        diversity_score=diversity,
        detail={
            "recovery_threshold": recovery_threshold,
            "transfer_threshold": transfer_threshold,
            "diversity_threshold": diversity_threshold,
            "high_omega": float(high_omega),
            "high_recovery": float(high_recovery),
            "good_transfer": float(good_transfer),
            "good_diversity": float(good_diversity),
        },
    )


def construct_degenerate_scenario(
    episodes_transitions: dict[str, list[StepTransition]],
    scenario: str = "reward_collapse",
) -> dict[str, list[StepTransition]]:
    """Construct a degenerate scenario from baseline episodes.

    Scenarios:
    - reward_collapse: Replace all rewards with the same value
    - stress_convergence: Add identical noise to all controllers
    - compression_narrowing: Remove lowest-performing controllers
    """
    result: dict[str, list[StepTransition]] = {}

    if scenario == "reward_collapse":
        # All controllers get the same reward trajectory
        all_rewards = []
        for name, trans in episodes_transitions.items():
            all_rewards.extend([t.reward for t in trans])
        mean_reward = float(np.mean(all_rewards)) if all_rewards else 0.0

        for name, trans in episodes_transitions.items():
            collapsed = []
            for t in trans:
                new_t = StepTransition(
                    observation=t.observation,
                    action=t.action,
                    reward=mean_reward,
                    terminated=t.terminated,
                    truncated=t.truncated,
                    info=t.info,
                )
                collapsed.append(new_t)
            result[name] = collapsed

    elif scenario == "stress_convergence":
        # Add the same perturbation to all controllers' observations
        rng = np.random.default_rng(42)
        for name, trans in episodes_transitions.items():
            perturbed = []
            for t in trans:
                new_t = StepTransition(
                    observation=t.observation,
                    action=t.action,
                    reward=t.reward + rng.normal(0, 0.1),
                    terminated=t.terminated,
                    truncated=t.truncated,
                    info=t.info,
                )
                perturbed.append(new_t)
            result[name] = perturbed

    elif scenario == "compression_narrowing":
        # Keep only the top-performing controller, duplicate it
        best_name = max(
            episodes_transitions.keys(),
            key=lambda n: sum(t.reward for t in episodes_transitions[n]),
        )
        for name in episodes_transitions:
            result[name] = list(episodes_transitions[best_name])

    else:
        result = dict(episodes_transitions)

    return result
