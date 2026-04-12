"""Claims ledger — structured tracking of experimental claims at three tiers.

Every major finding must be registered here with its evidence level:
- **useful**: beats a simpler baseline, survives at least one ablation, reproducible across seeds.
- **strong**: survives multiple task types, survives controller/world swaps, interpretable failure modes documented.
- **promoted**: cross-condition robust, negative controls fail correctly, not explainable by one trivial shortcut.

This module provides programmatic claim registration, validation, and
markdown export so that the ledger can be maintained alongside experiment
code and automatically included in reports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ClaimTier(str, Enum):
    USEFUL = "useful"
    STRONG = "strong"
    PROMOTED = "promoted"


# Forbidden phrases — must never appear in any claim text.
_FORBIDDEN_PHRASES = frozenset(
    {
        "the fly is conscious",
        "we found consciousness",
        "this proves awareness",
        "proves consciousness",
        "proves sentience",
        "proves phenomenology",
        "is aware",
        "is sentient",
        "has consciousness",
    }
)

# Approved vocabulary for claims about cognition-adjacent findings.
APPROVED_VOCABULARY = (
    "candidate prerequisite",
    "persistent integrated state",
    "observer-like organization proxy",
    "self/world separation marker",
    "control integration marker",
    "recurrent stabilization marker",
    "history dependence marker",
    "controller-invariant representation",
)


@dataclass(slots=True)
class Claim:
    """A single registered claim."""

    claim_id: str
    text: str
    tier: ClaimTier
    experiment: str
    evidence: list[str]
    ablation_survived: list[str] = field(default_factory=list)
    negative_controls: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    promoted_from: ClaimTier | None = None


@dataclass
class ClaimsLedger:
    """Registry of all experimental claims with validation and export."""

    claims: list[Claim] = field(default_factory=list)
    _id_counter: int = field(default=0, init=False)

    def register(
        self,
        text: str,
        tier: ClaimTier,
        experiment: str,
        evidence: list[str],
        *,
        ablation_survived: list[str] | None = None,
        negative_controls: list[str] | None = None,
        failure_modes: list[str] | None = None,
        caveats: list[str] | None = None,
    ) -> Claim:
        """Register a new claim after validation."""
        # Overclaiming filter.
        violations = overclaiming_filter(text)
        if violations:
            raise ValueError(
                f"Claim text contains forbidden language: {violations}"
            )

        # Tier-specific validation.
        if tier == ClaimTier.USEFUL:
            if len(evidence) < 1:
                raise ValueError("'useful' claims need at least one evidence item.")
        elif tier == ClaimTier.STRONG:
            if not ablation_survived:
                raise ValueError(
                    "'strong' claims must document survived ablations."
                )
            if not failure_modes:
                raise ValueError(
                    "'strong' claims must document failure modes."
                )
        elif tier == ClaimTier.PROMOTED:
            if not negative_controls:
                raise ValueError(
                    "'promoted' claims must include negative control results."
                )
            if not ablation_survived or len(ablation_survived) < 2:
                raise ValueError(
                    "'promoted' claims must survive multiple ablation families."
                )

        self._id_counter += 1
        claim = Claim(
            claim_id=f"CLM-{self._id_counter:04d}",
            text=text,
            tier=tier,
            experiment=experiment,
            evidence=evidence,
            ablation_survived=ablation_survived or [],
            negative_controls=negative_controls or [],
            failure_modes=failure_modes or [],
            caveats=caveats or [],
        )
        self.claims.append(claim)
        return claim

    def promote(self, claim_id: str, new_tier: ClaimTier) -> Claim:
        """Promote a claim to a higher tier after additional validation."""
        claim = self.get(claim_id)
        if claim is None:
            raise KeyError(f"Claim {claim_id} not found.")
        tier_order = {ClaimTier.USEFUL: 0, ClaimTier.STRONG: 1, ClaimTier.PROMOTED: 2}
        if tier_order[new_tier] <= tier_order[claim.tier]:
            raise ValueError(
                f"Cannot demote or keep same tier: {claim.tier} → {new_tier}"
            )
        claim.promoted_from = claim.tier
        claim.tier = new_tier
        return claim

    def get(self, claim_id: str) -> Claim | None:
        for c in self.claims:
            if c.claim_id == claim_id:
                return c
        return None

    def filter_by_tier(self, tier: ClaimTier) -> list[Claim]:
        return [c for c in self.claims if c.tier == tier]

    def filter_by_experiment(self, experiment: str) -> list[Claim]:
        return [c for c in self.claims if c.experiment == experiment]

    def to_markdown(self) -> str:
        """Export the full ledger as markdown."""
        lines = [
            "# Claims Ledger",
            "",
            "## Summary",
            "",
            f"- Total claims: {len(self.claims)}",
            f"- Useful: {len(self.filter_by_tier(ClaimTier.USEFUL))}",
            f"- Strong: {len(self.filter_by_tier(ClaimTier.STRONG))}",
            f"- Promoted: {len(self.filter_by_tier(ClaimTier.PROMOTED))}",
            "",
        ]

        for tier in (ClaimTier.PROMOTED, ClaimTier.STRONG, ClaimTier.USEFUL):
            tier_claims = self.filter_by_tier(tier)
            if not tier_claims:
                continue
            lines.append(f"## {tier.value.upper()} claims")
            lines.append("")
            for c in tier_claims:
                lines.append(f"### {c.claim_id}: {c.text}")
                lines.append("")
                lines.append(f"**Experiment**: {c.experiment}")
                lines.append("")
                lines.append("**Evidence**:")
                for e in c.evidence:
                    lines.append(f"- {e}")
                if c.ablation_survived:
                    lines.append("")
                    lines.append("**Ablations survived**:")
                    for a in c.ablation_survived:
                        lines.append(f"- {a}")
                if c.negative_controls:
                    lines.append("")
                    lines.append("**Negative controls**:")
                    for n in c.negative_controls:
                        lines.append(f"- {n}")
                if c.failure_modes:
                    lines.append("")
                    lines.append("**Failure modes**:")
                    for f in c.failure_modes:
                        lines.append(f"- {f}")
                if c.caveats:
                    lines.append("")
                    lines.append("**Caveats**:")
                    for cv in c.caveats:
                        lines.append(f"- {cv}")
                lines.append("")

        return "\n".join(lines)

    def to_json(self) -> str:
        """Export ledger as JSON."""
        data = []
        for c in self.claims:
            data.append(
                {
                    "claim_id": c.claim_id,
                    "text": c.text,
                    "tier": c.tier.value,
                    "experiment": c.experiment,
                    "evidence": c.evidence,
                    "ablation_survived": c.ablation_survived,
                    "negative_controls": c.negative_controls,
                    "failure_modes": c.failure_modes,
                    "caveats": c.caveats,
                    "promoted_from": c.promoted_from.value if c.promoted_from else None,
                }
            )
        return json.dumps(data, indent=2)

    def save(self, path: str | Path) -> None:
        """Save ledger as markdown and JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        md_path = path.with_suffix(".md")
        json_path = path.with_suffix(".json")
        md_path.write_text(self.to_markdown())
        json_path.write_text(self.to_json())


def overclaiming_filter(text: str) -> list[str]:
    """Check text for forbidden consciousness claims.

    Returns a list of violations found (empty if clean).
    """
    text_lower = text.lower()
    violations: list[str] = []
    for phrase in _FORBIDDEN_PHRASES:
        if phrase in text_lower:
            violations.append(phrase)
    return violations


def validate_claim_text(text: str) -> dict[str, Any]:
    """Validate claim text against overclaiming rules and vocabulary.

    Returns a dict with 'valid', 'violations', and 'suggestions'.
    """
    violations = overclaiming_filter(text)
    text_lower = text.lower()
    uses_approved = any(term in text_lower for term in APPROVED_VOCABULARY)
    return {
        "valid": len(violations) == 0,
        "violations": violations,
        "uses_approved_vocabulary": uses_approved,
        "suggestions": (
            []
            if uses_approved
            else [
                f"Consider using approved terms: {', '.join(APPROVED_VOCABULARY[:3])}, ..."
            ]
        ),
    }
