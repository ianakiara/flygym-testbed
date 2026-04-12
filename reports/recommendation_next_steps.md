# Recommendation & Next Steps

## Current Status After Deep Audit

**9 passed, 1 blocked.** Six bugs found and fixed. Three findings promoted.

The first real embodied validation round confirms that feedback, translation, and seams are structural laws worth keeping.

---

## What to Build Next

### Priority 1: MuJoCo/EGL Body Substrate Validation (CRITICAL)

The single most important gap is that **body substrate value has never been tested**. All 10 stages ran with BodylessBodyLayer.

**Action items:**
1. Set up a machine with MuJoCo and EGL/GPU support
2. Run Stages 3, 4, 6, 8 with FlyBodyLayer instead of BodylessBodyLayer
3. Compare: does the real body produce different stability, persistence, self/world metrics?
4. If body-preserving beats bodyless on ≥3/4 structure metrics in ≥2 worlds → body substrate claim is validated
5. If not → the body hypothesis is disproved, and the architecture should focus on the controller layer only

**Why this matters**: The body substrate hypothesis is the foundational claim of the research program. Without testing it, 40% of the architecture is unjustified. This is one of the five crown-jewel validations.

### Priority 2: Build Harsher Stage 6 Perturbation Protocol

Self/world signal exists (memory 0.077 vs bodyless 0.061) but is weak. The perturbation protocol needs to be adversarial.

**Action items:**
1. Add explicit perturbation families:
   - Self-motion perturbation only
   - World-motion perturbation only
   - Both together
   - False sensory motion (inject body oscillation)
   - Stabilization removed
   - Delayed proprioception
2. Increase external event frequency (every 2-3 steps instead of 7)
3. Increase noise scale (0.1-0.2 instead of 0.02)
4. Measure compensation latency and confusion rate separately (not just correlation)
5. Re-run Stage 6 — it should feel uncomfortable and adversarial

### Priority 3: Upgrade Stage 9 to BackboneShared-Style Metric

The current proxy (target distance CV + persistence) cannot distinguish genuine shared objectness from collapse-induced similarity.

**Action items:**
1. Implement the fuller composite score:
   ```
   BackboneShared(X) = Ω(X) - λ·InteropLoss(X) - μ·SeamFragility(X) - ν·ScaleDrift(X)
   ```
2. This is feasible NOW because we already have interop signal, seam signal, and multi-world transfer
3. Track at least three object types: target in world, goal state bundle, internal latent cluster
4. Run stressors: controller swap, world change, sensory dropout, history manipulation
5. Re-run Stage 9

### Priority 4: RL Controller for Interoperability Stress Test

All current controllers are hand-designed. Testing with a learned (RL) controller would strengthen CLM-002.

**Action items:**
1. Train a simple RL controller (PPO or SAC) on the avatar world
2. Add it to the interoperability comparison
3. Test: does translated alignment still beat raw alignment when one controller is learned?
4. If yes → interoperability finding is very strong
5. If no → interoperability may be an artifact of hand-designed controller similarity

### Priority 5: Cross-World Interoperability

Stage 7 only tested interoperability within one world. Cross-world testing would strengthen or falsify the finding.

**Action items:**
1. Run each controller pair in simplified, avatar, and native worlds
2. Compute interoperability across worlds (same controller, different worlds)
3. Test: does shared task structure survive world changes?

---

## What to Stop

### Stop 1: Do not claim body substrate value
The hypothesis is untested (blocked, not failed). Do not claim the body matters until FlyBodyLayer tests have been run. It may turn out that the body DOESN'T matter, and that would be a valuable finding too.

### Stop 2: Do not over-interpret the pose channel finding
While pose ablation causes 100% stability collapse, this is partly because the stability metric IS computed from pose features. The causal direction needs to be tested with behavioral outcomes (does the agent actually fail?), not just metric values.

### Stop 3: Do not claim strong self/world separation yet
The signal (0.077 vs 0.061) is real but small. Need harsher perturbation protocol before promoting.

### Stop 4: Do not claim shared objectness with current proxy
The proxy is insufficient. Stress-induced similarity is a collapse artifact, not genuine shared representation.

---

## What to Port to the Broader Research Stack

### 1. Seam Fragility Law (HIGHEST TRANSFER VALUE)

**Finding**: Broken seams cause system-level failures that local module metrics miss. Reward can improve while the system structurally collapses.

**Applicable to**:
- Any modular robotics system
- Multi-agent systems with communication channels
- Microservice architectures with API contracts
- Distributed ML systems with model-to-model handoffs

**How to port**:
- Instrument all module-to-module interfaces
- Measure downstream stability (not just local performance) under seam perturbation
- If breaking an interface improves local metrics but degrades global stability → the seam is structurally critical

### 2. Controller Interoperability via Task Structure

**Finding**: Different controller families converge on similar solutions through task constraints, not shared representations.

**Applicable to**:
- Multi-agent policy comparison
- Transfer learning evaluation
- Ensemble methods
- Human-AI team integration

**How to port**:
- Compare controllers by translated (outcome-space) alignment, not raw latent correlation
- Translated alignment captures shared task structure even when internal representations differ completely

### 3. Channel-Group Ablation Methodology

**Finding**: Per-channel-group ablation reveals which feedback channels are structurally critical vs. redundant.

**Applicable to**:
- Sensor suite design (which sensors actually matter?)
- Feature importance in production ML systems
- Communication channel prioritization in distributed systems

**How to port**:
- Group related channels
- Zero each group independently
- Measure stability (not just performance) under ablation
- Channels that cause stability collapse without performance drop are structurally critical

---

## Timeline Recommendation

| Priority | Effort | Impact | Timeline |
|----------|--------|--------|----------|
| P1: MuJoCo body testing | High | Critical | 1-2 weeks |
| P2: Harsher self/world protocol | Medium | High | 3-5 days |
| P3: BackboneShared metric | Medium | High | 3-5 days |
| P4: RL controller | High | Medium | 1-2 weeks |
| P5: Cross-world interop | Low | Medium | 2-3 days |

**Recommended order**: P2 → P3 → P5 → P1 → P4

Start with the protocol/metric upgrades (P2, P3) because they may strengthen Stage 6 and Stage 9 results. Then do cross-world interop (P5) to stress-test the strongest finding. Then tackle MuJoCo (P1) which requires infrastructure changes. RL controller (P4) is last because it depends on training infrastructure.

---

## One-Line Summary

Your first real embodied validation round already confirms that feedback, translation, and seams are structural laws worth keeping — and the failures are honest enough to trust the passes.
