# Claims Ledger

## Purpose

This ledger tracks all experimental claims at three validation tiers. Every claim must be registered here with its evidence, ablation results, and negative controls. Claims are promoted only when they meet increasingly stringent criteria.

## Validation tiers

### Useful
- Beats a simpler baseline
- Survives at least one ablation family
- Reproducible across seeds (5 seeds minimum)

### Strong
- Survives multiple task types
- Survives controller/world swaps
- Interpretable failure modes documented

### Promoted
- Cross-condition robust
- Negative controls fail correctly
- Not explainable by one trivial shortcut

## Approved vocabulary

All claims must use engineering/behaviour framing. Approved terms:
- candidate prerequisite
- persistent integrated state
- observer-like organization proxy
- self/world separation marker
- control integration marker
- recurrent stabilization marker
- history dependence marker
- controller-invariant representation

## Forbidden language

The following phrases must NEVER appear in any claim:
- "the fly is conscious"
- "we found consciousness"
- "this proves awareness"
- "proves consciousness / sentience / phenomenology"
- "is aware" / "is sentient" / "has consciousness"

## Implementation

The claims ledger is implemented programmatically in `flygym_research.cognition.validation.claims_ledger`. Claims are registered, validated against overclaiming rules, and exported as markdown/JSON.

```python
from flygym_research.cognition.validation import ClaimsLedger, ClaimTier

ledger = ClaimsLedger()

# Register a "useful" claim
ledger.register(
    text="Body substrate provides higher state persistence (candidate prerequisite for integrated control)",
    tier=ClaimTier.USEFUL,
    experiment="Body Substrate Value",
    evidence=["Embodied agents show 2x higher state autocorrelation than bodyless agents across 5 seeds"],
)

# Register a "strong" claim
ledger.register(
    text="Ascending stability feedback is a candidate prerequisite for coherent higher-level control",
    tier=ClaimTier.STRONG,
    experiment="Ascending Feedback Ablation",
    evidence=["Stability ablation degrades task performance by 40%", "Effect replicates across navigation and tracking tasks"],
    ablation_survived=["Contact ablation", "Effort ablation"],
    failure_modes=["Performance recovers partially when memory controller compensates for missing stability signal"],
)

# Export
ledger.save("reports/claims_ledger")
```

## Current claims

*To be populated as experiments are run. Use `ledger.to_markdown()` for the full formatted output.*

### Summary

| Tier | Count | Description |
| --- | --- | --- |
| Promoted | 0 | Cross-condition robust, negative controls verified |
| Strong | 0 | Multi-task, multi-condition, failure modes documented |
| Useful | 0 | Beats baseline, survives ablation, reproducible |

## Process

1. Run experiment → collect metrics
2. Compare against baselines and ablations
3. Register claim at appropriate tier
4. Document evidence, failures, and caveats
5. Promote claims as additional evidence accumulates
6. Export ledger for reports

## Review checklist

Before promoting any claim:
- [ ] Beats simpler baseline (quantified)
- [ ] Survives at least one ablation family
- [ ] Reproducible across 5+ seeds
- [ ] Negative control (random/shuffled) performs worse
- [ ] Failure modes documented and interpretable
- [ ] Not explainable by one trivial shortcut
- [ ] Uses approved vocabulary only
- [ ] No consciousness/sentience claims
