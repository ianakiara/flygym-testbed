# True vs Hype

## Real Engineering Truths

### 1. The three-layer architecture (Brain/Body/World) works as designed
- All 8 controllers run successfully in all 3 world modes
- Deterministic seeding produces identical rollouts
- Ablations toggle cleanly by config with no code changes required
- Observation/action shapes are consistent across all configurations
- **This is solid engineering, not hype.**

### 2. Ascending feedback channels have a measurable structural hierarchy
- Pose channel ablation: 100% stability collapse
- Locomotion/contact/target/internal ablation: 0% stability effect individually
- This is a real, reproducible, falsifiable finding across 10 seeds with zero variance
- **Caveat**: In BodylessBodyLayer, "stability" is computed from pose features — so this is partly circular. Needs MuJoCo validation.

### 3. Controllers share task structure even when behavior differs
- Translated interoperability score: 0.437 mean across 10 pairs
- Raw latent correlation: 0.000 across all pairs
- Controllers that solve the same task develop similar reward trajectories and action distributions even with completely different internal mechanisms
- **This is a genuine structural finding.**

### 4. Seam perturbation reveals failures that local metrics miss
- Breaking pose+locomotion channels: reward improves by 26.05, but stability drops 100%
- A system that looks "better" by reward is actually structurally broken
- **This is the most practically valuable finding** — it directly applies to any modular system.

### 5. History dependence is real (after metric correction)
- Memory controller: within-bucket temporal autocorrelation = 0.818
- Random controller: 0.034
- The old variance-based metric was inverted (random scored highest). The corrected autocorrelation-based metric correctly shows memory controllers produce temporally structured actions in the same coarse state.
- **This is a real signal, but needs stronger validation** (matched-state predictors).

### 6. Self/world disambiguation signal exists (after metric correction)
- Memory controller |self_world| = 0.077, beats all baselines (bodyless=0.061, random=0.005, reflex=0.000)
- The old metric used body_speed (disconnected from world events) and let static controllers trivially score 1.0
- **Signal is real but weak** — needs harsher perturbation protocol to strengthen.

---

## Real Research Insights

### 1. Channel group architecture matters for stability
- Not all feedback channels are equal — pose (containing stability, height, speed) is critical
- Other channels (locomotion quality, contact, target, phase) are structurally redundant for stability
- This suggests a priority hierarchy in ascending feedback that mirrors biological sensory processing

### 2. Interoperability is achievable through task structure
- Different controller families (reflexive, memory-augmented, planning-based, raw) all converge on similar task solutions
- The convergence is in outcome space, not behavior space — controllers act differently but achieve similarly
- This suggests the task itself imposes structural constraints that any adequate controller must satisfy

### 3. Stress reveals structural degradation that metrics miss
- Pose ablation causes internal objectness to drop (0.656→0.484) while shared similarity rises (0.586→0.898)
- Controllers converge to similar broken behavior under stress — high shared score is a collapse artifact, not genuine coherence
- This is a real finding about how stress degrades complex systems

---

## Likely Overreads

### 1. "Body substrate improves structural control"
- **BLOCKED — NOT TESTED.** All tests ran with BodylessBodyLayer. The body substrate hypothesis is entirely untested.
- **Do not claim body substrate value until FlyBodyLayer tests run with MuJoCo.**

### 2. "Pose channel is THE critical channel"
- While pose ablation causes 100% stability collapse, this is partly because the stability metric is computed FROM pose features (stability, thorax_height_mm, body_speed_mm_s are all in the pose group)
- Zeroing the inputs to a metric and then measuring that metric is tautological
- **The finding is real but weaker than it appears.** Need to measure downstream behavioral effects, not just the metric that uses those inputs.

### 3. "Shared objectness proxy captures real objectness"
- The current proxy (target distance CV + persistence) is too simplistic
- Under stress, controllers degenerate to similar broken behavior — high shared scores are collapse artifacts
- **Proxy needs upgrade to BackboneShared-style metric before making objectness claims.**

---

## Invalid Interpretations

### 1. "The simulated fly shows proto-awareness"
- Nothing in these results supports any consciousness-adjacent claim
- We tested structural prerequisites for organized control, not experience

### 2. "The architecture proves embodied cognition"
- Without MuJoCo body physics, we have not tested embodiment at all
- The bodyless layer is a kinematic placeholder, not an embodied system

### 3. "10/10 stages validate the research program"
- 9 passed, 1 blocked. The blocked stage (body substrate) is one of the five crown-jewel validations.
- Several passing stages have caveats and need stronger protocols.

---

## Summary Table

| Claim | Verdict | Confidence |
|-------|---------|------------|
| Architecture works | **TRUE** | High |
| Ascending loop matters | **TRUE (with caveat)** | Medium-High |
| Controller interop is real | **TRUE** | High |
| Seam law holds | **TRUE** | High |
| History dependence | **TRUE (after metric fix)** | Medium |
| Self/world separation | **WEAK BUT REAL** | Low-Medium |
| Stability transfers | **TRUE** | Medium |
| Stress reveals degradation | **TRUE** | Medium |
| Body substrate matters | **BLOCKED** | N/A |
| Shared objectness proxy | **INSUFFICIENT** | Low |
