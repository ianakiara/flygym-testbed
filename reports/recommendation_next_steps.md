# Recommendation & Next Steps

## What to Build Next

### Priority 1: MuJoCo/EGL Body Substrate Validation (CRITICAL)

The single most important gap is that **body substrate value has never been tested**. All 10 stages ran with BodylessBodyLayer.

**Action items:**
1. Set up a machine with MuJoCo and EGL/GPU support
2. Run Stages 3, 4, 6, 8 with FlyBodyLayer instead of BodylessBodyLayer
3. Compare: does the real body produce different stability, persistence, self/world metrics?
4. If body-preserving beats bodyless on ≥3/4 structure metrics in ≥2 worlds → body substrate claim is validated
5. If not → the body hypothesis is disproved, and the architecture should focus on the controller layer only

**Why this matters**: The body substrate hypothesis is the foundational claim of the research program. Without testing it, 40% of the architecture is unjustified.

### Priority 2: Fix the History Dependence Metric

The current metric (action variance in state buckets) is fundamentally flawed — it rewards randomness.

**Action items:**
1. Implement prediction-based history dependence:
   - Train a state-only MLP predictor on (current_state → next_action)
   - Train a history-aware LSTM/transformer predictor on (state_history → next_action)
   - Compare AUC on held-out episodes
2. Build matched-state dataset: episodes where current observation is identical but prior history differs
3. Measure: does knowing history improve next-regime prediction? repair cost prediction?
4. Re-run Stage 5 with the new metric

### Priority 3: Strengthen Self/World Perturbation Protocol

The current self/world separation test produces no signal because perturbations are too subtle.

**Action items:**
1. Increase external event frequency (every 2-3 steps instead of 7)
2. Increase noise scale (0.1-0.2 instead of 0.02)
3. Add explicit self-motion perturbation (inject body oscillation)
4. Add explicit world-motion perturbation (shift target position)
5. Measure compensation latency and confusion rate separately
6. Re-run Stage 6

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

### Stop 1: Do not claim history dependence based on current metrics
The variance-based metric is an artifact. Any claim about history dependence is currently unsupported. Remove these claims from any research communications until the metric is fixed and re-validated.

### Stop 2: Do not claim self/world separation
The signal is indistinguishable from noise (0.029 vs 0.029). Do not cite self_world_marker scores as evidence for anything until the perturbation protocol is fixed.

### Stop 3: Do not claim shared objectness
The gap (0.001) is negligible. The objectness proxy is too simplistic. Do not promote shared objectness as a finding until proper learned representations and cross-controller object tracking are implemented.

### Stop 4: Do not claim body substrate value
The hypothesis is untested. Do not claim the body matters until FlyBodyLayer tests have been run. It may turn out that the body DOESN'T matter, and that would be a valuable finding too.

### Stop 5: Do not over-interpret the pose channel finding
While pose ablation causes 100% stability collapse, this is partly because the stability metric IS computed from pose features. The causal direction needs to be tested with behavioral outcomes (does the agent actually fail?), not just metric values.

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
| P2: History metric fix | Medium | High | 3-5 days |
| P3: Self/world perturbation | Medium | High | 3-5 days |
| P4: RL controller | High | Medium | 1-2 weeks |
| P5: Cross-world interop | Low | Medium | 2-3 days |

**Recommended order**: P2 → P3 → P5 → P1 → P4

Start with the quick metric fixes (P2, P3) because they may flip Stage 5 and Stage 6 results. Then do cross-world interop (P5) to strengthen the strongest finding. Then tackle MuJoCo (P1) which requires infrastructure changes. RL controller (P4) is last because it depends on training infrastructure.
