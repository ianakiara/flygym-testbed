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

---

## Likely Overreads

### 1. "Body substrate improves structural control"
- **NOT VALIDATED.** All tests ran with BodylessBodyLayer. The body substrate hypothesis is entirely untested.
- The fact that BodylessAvatarController and ReducedDescendingController produce identical results in bodyless mode does NOT mean the body doesn't matter — it means we haven't tested it.
- **Do not claim body substrate value until FlyBodyLayer tests run with MuJoCo.**

### 2. "History dependence is structurally real"
- **ARTIFACT.** The current metric (action variance in state buckets) is maximized by random noise, not meaningful path dependence.
- Memory controller (0.008) shows LESS "history dependence" than random (0.291)
- This is a metric design failure, not evidence for or against history dependence
- **Do not cite these numbers as evidence of anything.**

### 3. "Pose channel is THE critical channel"
- While pose ablation causes 100% stability collapse, this is partly because the stability metric is computed FROM pose features (stability, thorax_height_mm, body_speed_mm_s are all in the pose group)
- Zeroing the inputs to a metric and then measuring that metric is tautological
- **The finding is real but weaker than it appears.** Need to measure downstream behavioral effects, not just the metric that uses those inputs.

### 4. "Shared objectness is real"
- Gap of 0.001 between shared and internal objectness is statistically insignificant
- The objectness proxy is too simplistic (target distance CV + persistence)
- **This is conjecture, not a finding.**

---

## Invalid Interpretations

### 1. "The simulated fly shows proto-awareness"
- Nothing in these results supports any consciousness-adjacent claim
- We tested structural prerequisites for organized control, not experience

### 2. "Ascending feedback enables self/world separation"  
- self_world_marker scores are indistinguishable between structured controllers (0.029) and random baselines (0.029)
- The metric and environment are too weak to test this claim

### 3. "The architecture proves embodied cognition"
- Without MuJoCo body physics, we have not tested embodiment at all
- The bodyless layer is a kinematic placeholder, not an embodied system

### 4. "10/10 stages validate the research program"
- 6/10 passed, 4/10 failed. The failures are real and meaningful.
- The 4 failures identify specific gaps: body substrate testing, metric design, perturbation strength, objectness measurement.

---

## Summary Table

| Claim | Verdict | Confidence |
|-------|---------|------------|
| Architecture works | **TRUE** | High |
| Ascending loop matters | **TRUE (with caveat)** | Medium-High |
| Controller interop is real | **TRUE** | High |
| Seam law holds | **TRUE** | High |
| Stability transfers | **TRUE** | Medium |
| Body substrate matters | **UNTESTED** | N/A |
| History dependence | **ARTIFACT** | High (that it's wrong) |
| Self/world separation | **WEAK** | Low |
| Shared objectness | **WEAK** | Low |
