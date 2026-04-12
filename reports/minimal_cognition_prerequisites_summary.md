# Minimal Cognition Prerequisites Summary

## Purpose

This report summarises which candidate prerequisites for minimal embodied cognition are supported, refuted, or unresolved based on the experimental evidence from this project.

## Framing

We do NOT claim to have found consciousness, sentience, or phenomenology. We investigate **candidate prerequisites** — measurable properties that appear necessary (but not sufficient) for systems that might eventually exhibit observer-like organisation.

All findings are framed as:
- **engineering claims** (the architecture supports X)
- **behaviour/control claims** (the controller exhibits Y)
- **systems claims** (the system has property Z)
- **candidate prerequisite claims** (Z is a candidate prerequisite for minimal cognition)

## Candidate prerequisites investigated

### 1. Persistent integrated state

**Definition**: The controller maintains internal state that is informative across time steps and integrates multiple information sources.

**Metrics**: State persistence (autocorrelation), cross-time MI, state decay curves, predictive utility.

**Experiments**: 1 (body substrate value), 5 (history vs state).

**Status**: *Pending experimental results.*

**Assessment criteria**:
- TRUE if memory controller shows significantly higher persistence than memoryless controllers
- TRUE if state decay half-life > 5 steps for memory controller
- FALSE if random controller shows comparable persistence

### 2. History dependence

**Definition**: Same current observation leads to different actions depending on past trajectory.

**Metrics**: History dependence (action variance in coarse-state buckets), hysteresis score.

**Experiments**: 5 (history vs state), with HistoryDependenceTask.

**Status**: *Pending experimental results.*

**Assessment criteria**:
- TRUE if memory controller's hysteresis score > 2x reduced descending controller
- FALSE if random controller shows non-zero hysteresis

### 3. Self/world disambiguation

**Definition**: The controller distinguishes self-induced sensory changes from externally caused ones.

**Metrics**: Self/world separation (event↔speed correlation).

**Experiments**: 3 (feedback ablation), with SelfWorldDisambiguationTask.

**Status**: *Pending experimental results.*

**Assessment criteria**:
- TRUE if ablating stability feedback degrades self/world metric more than other ablations
- FALSE if metric is unchanged across ablation conditions

### 4. Stabilisation as prerequisite

**Definition**: Body/reflex stabilisation is necessary for coherent higher-level control.

**Metrics**: Stabilisation quality, task performance as function of stabilisation gain.

**Experiments**: 4 (stabilisation dependence).

**Status**: *Pending experimental results.*

**Assessment criteria**:
- TRUE if there's a dose-response relationship between stabilisation gain and higher-level performance
- FALSE if performance is unchanged across gain levels

### 5. Controller interoperability

**Definition**: Different controller families produce translatable representations when operating on the same body/world.

**Metrics**: Interoperability score, latent correlation, action distribution similarity.

**Experiments**: 2 (controller swap).

**Status**: *Pending experimental results.*

**Assessment criteria**:
- TRUE if memory and planner controllers show high mutual latent correlation
- FALSE if all pairs show equally low interoperability

### 6. Body substrate value

**Definition**: The physical body/reflex substrate adds measurable value beyond what a bodyless agent achieves.

**Metrics**: All core metrics compared between embodied and bodyless conditions.

**Experiments**: 1 (body substrate value).

**Status**: *Pending experimental results.*

**Assessment criteria**:
- TRUE if embodied agents outperform bodyless on persistence and self/world metrics
- FALSE if bodyless agents match embodied on all metrics

### 7. Cross-world coherence

**Definition**: Higher controllers retain organisation when the world changes but the body substrate is preserved.

**Metrics**: Cross-world metric variance, target representation stability.

**Experiments**: 6 (avatar-world transfer), 8 (shared target test).

**Status**: *Pending experimental results.*

**Assessment criteria**:
- TRUE if within-controller cross-world variance is low for memory/planner controllers
- FALSE if all controllers show high cross-world variance

## Summary table

| Prerequisite | Status | Key experiment | Key metric |
| --- | --- | --- | --- |
| Persistent integrated state | Pending | Exp 1, 5 | State persistence, MI |
| History dependence | Pending | Exp 5 | Hysteresis score |
| Self/world disambiguation | Pending | Exp 3 | Self/world marker |
| Stabilisation prerequisite | Pending | Exp 4 | Stab quality vs performance |
| Controller interoperability | Pending | Exp 2 | Interoperability score |
| Body substrate value | Pending | Exp 1 | Embodied vs bodyless delta |
| Cross-world coherence | Pending | Exp 6, 8 | Cross-world variance |

## What this project CAN and CANNOT tell us

### CAN
- Whether the architecture supports clean experimentation on these prerequisites
- Whether specific feedback channels, body properties, or memory architectures measurably affect these metrics
- Which prerequisites survive ablation and which are trivially explained

### CANNOT
- Whether any of these prerequisites are sufficient for consciousness
- Whether the fly "experiences" anything
- Whether these findings generalise beyond the FlyGym substrate

## Next steps

1. Run all 8 experiments and populate this table
2. Promote validated findings to claims ledger
3. Identify which prerequisites form a minimal set vs. which are redundant
4. Design second-generation experiments targeting the most promising prerequisites
5. Compare with other embodied cognition frameworks (e.g., OpenAI Gym robotics, DeepMind control suite)
