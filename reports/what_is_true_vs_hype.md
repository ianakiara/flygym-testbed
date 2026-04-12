# What Is True vs Hype

## Purpose

This report separates experimentally validated findings from unvalidated claims, speculative interpretations, and common overclaims in embodied cognition research. It is the falsification-first summary of the project.

## Framework

Every finding is classified into one of four categories:

| Category | Criteria | Action |
| --- | --- | --- |
| **TRUE** | Survives ablation, beats baselines, reproduces across seeds | Carry forward |
| **LIKELY** | Beats baseline but hasn't survived all ablation families | Investigate further |
| **UNCERTAIN** | Mixed results or confounded design | Do not cite without caveats |
| **HYPE** | Not supported by evidence, overclaimed, or trivially explained | Retract or reframe |

## Architecture claims

### Three-layer separation (Brain/Body/World)

**Status**: TRUE (architectural — not an empirical claim)

The architecture cleanly separates concerns:
- Brain controllers can be swapped without changing body or world
- World modes can be swapped while preserving body/reflex substrate
- Ascending and descending interfaces decouple layers

This is a design property, not an empirical discovery. It enables experiments but doesn't prove anything about cognition.

### Reduced descending interface

**Status**: TRUE (engineering claim)

The reduced descending interface maps high-level commands to body-level actuator patterns. This is testable and verified by the descending adapter tests.

**Not proven**: Whether reduced control is *better* than raw control for any particular objective. This is an empirical question addressed by Experiment 7.

## Body substrate claims

### "The body substrate provides value for higher-level control"

**Status**: UNCERTAIN pending Experiment 1 results

**What would make it TRUE**: Embodied agents consistently outperform bodyless agents on persistence, history dependence, and self/world separation metrics across multiple controllers and seeds.

**What would make it HYPE**: Bodyless agents match embodied agents, or gains are entirely attributable to the reflex-only controller (no higher-level benefit).

## Feedback channel claims

### "Ascending feedback is necessary for stable control"

**Status**: UNCERTAIN pending Experiment 3 results

**What would make it TRUE**: Ablating feedback channels degrades performance in a graded, interpretable way with stability being the most critical channel.

**What would make it HYPE**: Performance is unchanged after ablation, or all channels are equally dispensable.

## History dependence claims

### "Memory-augmented controllers show genuine history dependence"

**Status**: UNCERTAIN pending Experiment 5 results

**What would make it TRUE**: Memory controller shows significantly higher MI, slower decay, and higher hysteresis than memoryless controllers, with random controller at zero.

**What would make it HYPE**: All controllers show similar history metrics, or the random controller shows non-zero history dependence.

## Cross-world transfer claims

### "Controllers retain coherent organisation across worlds"

**Status**: UNCERTAIN pending Experiment 6 results

**What would make it TRUE**: Memory/planner controllers show consistent metric profiles across native, simplified, and avatar worlds.

**What would make it HYPE**: No controller transfers well, or all controllers (including random) show similar cross-world behaviour.

## Common overclaims to watch for

1. **"Performance = cognition"**: High task performance does not imply cognitive organisation. Random controllers can sometimes perform well on simple tasks.

2. **"Body = better"**: The body substrate is not automatically beneficial. It may add noise, slow learning, or provide no benefit for certain task types.

3. **"Memory = understanding"**: Having a memory buffer doesn't mean the controller uses it meaningfully. Check MI and predictive utility, not just buffer existence.

4. **"Stability = awareness"**: Stabilisation is a control property, not a consciousness marker. Frame it as a candidate prerequisite, not evidence of awareness.

5. **"Cross-world transfer = generalisation"**: Transferring across three similar worlds with the same body is not the same as general intelligence. The worlds share structure.

## Methodology for evaluating claims

1. State the claim precisely
2. Identify the simplest alternative explanation
3. Design an experiment that can falsify the claim
4. Run negative controls (random, shuffled, bodyless)
5. Check ablation survival
6. Check cross-condition robustness
7. Document failure modes
8. Register in claims ledger at appropriate tier

## Conclusion

*To be updated as experiments are completed.*

At this stage, the project has built the infrastructure for falsifiable experimentation. No empirical claims have been promoted yet. The architecture itself is validated as a clean experimental substrate. All cognition-adjacent claims remain UNCERTAIN until experimental evidence is collected and validated against the three-tier standard.
