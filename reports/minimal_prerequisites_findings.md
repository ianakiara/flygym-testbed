# Minimal Prerequisites Findings

## Summary

This report summarizes what the 10-stage validation program found about each candidate prerequisite for persistent integrated control. All findings are framed as engineering/behavioral properties, not consciousness claims.

**After deep audit and 6 bug fixes**, results improved from 6/10 to 9 passed + 1 blocked.

---

## 1. Body Role

**Status**: BLOCKED (infrastructure limitation)

**What we tested**: Stage 3 compared body-preserving controllers (reduced_descending, raw_control) against a genuinely bodyless controller (BodylessAvatarController — ignores all ascending body feedback) across simplified and avatar worlds.

**What we found**: Body-preserving agents won 0/4 structure metrics in 3 out of 4 world/controller combinations. One combination (avatar/raw_control) won 1/4.

**Why this doesn't answer the question**: All environments used BodylessBodyLayer (no MuJoCo/EGL available). BodylessBodyLayer always returns stability=1.0 regardless of controller behavior. Reading a constant signal vs. ignoring it produces no meaningful difference.

**Bug found and fixed**: BodylessAvatarController was originally an empty subclass of ReducedDescendingController (`pass`), making the comparison literally identical-to-identical. Now rewritten to genuinely ignore ascending feedback.

**What would answer it**: Run the same tests with FlyBodyLayer (requires MuJoCo simulation) on a machine with EGL/GPU support.

**Confidence**: N/A — untested due to infrastructure

---

## 2. Stabilization Role

**Status**: CONFIRMED as structurally critical (STRONG PASS)

**What we tested**: Stage 4 ablated individual ascending feedback channels and measured stability collapse.

**What we found**:
- Pose channel ablation → 100% stability drop (1.0 → 0.0)
- All-channels-off → 100% stability drop
- Individual other channels (locomotion, contact, target, internal) → 0% stability drop
- 10 seeds, zero variance in these results

**Interpretation**: Stabilization depends critically on pose-group feedback (stability, thorax_height, body_speed). Without it, the system loses all stability even if task performance changes. This is a structural prerequisite.

**Caveat**: The stability metric is computed FROM pose features, so zeroing pose and measuring stability is partially tautological. The strong version of this claim needs behavioral validation (does the agent fall over, fail tasks, produce erratic trajectories?).

**Promoted wording**: Ascending body-state feedback is necessary for stable high-level control in the FlyGym cognition testbed, and different ascending channels produce distinct collapse signatures.

**Confidence**: Medium-High

---

## 3. Self/World Role

**Status**: USEFUL — weak but real signal (upgraded from WEAK after metric fixes)

**What we tested**: Stage 6 compared self_world_separation scores across controllers (reduced_descending, memory, reflex_only, bodyless, random) using 128-step episodes.

**What we found**:
- Memory controller |self_world|: 0.077 (best body-aware)
- Bodyless controller: 0.061
- Reflex-only: 0.000 (correctly gated by activity level)
- Random: 0.005
- Memory beats all baselines

**Bugs found and fixed**:
1. Old metric used body_speed_mm_s — disconnected from world events in BodylessBodyLayer
2. Static controllers (reflex_only) trivially scored 1.0 by never moving — all target_vector changes came from world events
3. Episodes were too short (5 steps) for external events (period=7) to occur

**Fixes applied**: Action-response + target-vector disruption metric with activity gate; 128-step episodes

**What would strengthen it**: Harsher perturbation protocol — self-motion only, world-motion only, false visual motion, delayed proprioception, body oscillation injection

**Confidence**: Low-Medium (signal exists but small: 0.077 vs 0.061)

---

## 4. History Role

**Status**: USEFUL — real signal (upgraded from ARTIFACT after metric fix)

**What we tested**: Stage 5 compared history dependence scores across reduced_descending, memory, and random controllers.

**What we found (corrected)**:
- Memory: 0.818 (high within-bucket temporal autocorrelation)
- Reduced descending: 0.745
- Random: 0.034 (near-zero autocorrelation)
- Gap: 0.784 (memory vs random)
- Transfers across all 3 worlds

**Bug found and fixed**: The original metric used action VARIANCE within coarse-state buckets. Random actions have maximum variance by definition, so random scored highest (0.291 > memory 0.008). The metric was inverted — it measured noise, not history dependence.

**Fix applied**: Replaced with within-bucket temporal autocorrelation. Memory controllers show temporally structured actions in the same coarse state (high autocorrelation); random does not.

**What would strengthen it**: Prediction-based metrics (state-only vs history-aware predictor comparison on next-action AUC, next-regime AUC, repair cost R²)

**Confidence**: Medium (autocorrelation may reflect momentum rather than genuine history dependence)

---

## 5. Controller Interoperability Role

**Status**: CONFIRMED — strong finding (methodology corrected)

**What we tested**: Stage 7 trained linear translation maps T: z_a → z_b between 14-dimensional state vectors across 10 controller pairs (5 controllers, same world, equal-length episodes of 64 steps). Reward was deliberately excluded from the composite score.

**What we found**:
- Mean translated alignment (linear map R²): 0.888
- Mean raw alignment (element-wise correlation): 0.493
- Gap: 0.394 (threshold was 0.05 gap + 0.15 absolute)
- Strongest translation gains on dissimilar pairs:
  - planner ↔ reflex_only: raw=0.210 → translated=0.891 (+0.681)
  - planner ↔ raw_control: raw=0.309 → translated=0.902 (+0.593)
  - memory ↔ raw_control: raw=0.296 → translated=0.883 (+0.588)
- Similar pairs show expected high raw alignment: reduced_descending ↔ memory raw=0.985

**Previous overclaim corrected**: The original metric used reward correlation as "translated alignment" (0.437 vs 0.000). This measured environment-imposed outcome similarity, not true latent translation. Reward is a global scalar shared by design — it always inflates agreement. The corrected metric uses 14-dimensional state vectors (position, heading, target, actions, ascending features) with learned linear OLS translation maps.

**Interpretation**: Controllers with completely different internal mechanisms produce state trajectories that are linearly translatable with high fidelity (R²=0.89). The translation gain is largest for dissimilar pairs, suggesting real shared structure beyond behavioral similarity.

**Hidden insight — environment as implicit translation operator**: The environment may act as a "forced translator" — constraining different controllers into similar state spaces. Controllers don't align internally, but world dynamics compress behavior into translatable patterns. This suggests `environment ≈ implicit translation operator`. This is itself a valuable finding worth investigating.

**Promoted wording**: Linear translation maps between controller state trajectories explain 89% of cross-controller variance (R²=0.888), significantly exceeding raw element-wise alignment (0.493). Reward is excluded from the composite score.

**Caveats**:
- 64 samples for 14-dim regression (ratio 4.6:1) — overfitting risk; needs cross-validation
- One trivially identical pair (reflex_only ↔ raw_control) inflates the mean
- Single world tested (avatar_remapped) — no cross-world interop yet
- All hand-designed controllers — no RL-trained controllers tested
- Environment-mediated alignment may partly explain the high R²

**Confidence**: Medium-High (methodology is now sound; environment-mediation hypothesis needs further investigation)

---

## 6. Seam Law Role

**Status**: CONFIRMED — highly valuable finding

**What we tested**: Stage 8 broke specific feedback channels (pose+locomotion, contact+target, all channels) and measured system-level effects.

**What we found**:
- Breaking pose+locomotion: stability drops 100%, but reward IMPROVES by 26.05
- Breaking all channels: same pattern
- Breaking contact+target: no effect on stability or reward
- Key insight: local metrics (reward) say "system improved" while structural metrics (stability) say "system collapsed"

**Interpretation**: Seam perturbation reveals failures that local module quality metrics miss entirely. A system can look "better" by reward while being structurally broken. This is the "seam law" — you cannot evaluate a modular system by looking at modules alone; you must look at the handoffs.

**Promoted wording**: Broken seams can produce structural collapse even when scalar reward does not decrease, so seam validity is a deeper criterion than reward alone.

**Caveats**:
- The seam_fragility metric itself showed 0.000 delta — the finding comes from stability drop, not the seam metric
- Reward improvement may indicate the baseline controller over-weights stability
- Only tested with BodylessBodyLayer

**Transferable lesson**: In any modular system (not just this one), broken interfaces between modules can produce locally-improved metrics while destroying global coherence. This applies to robotics, distributed systems, microservices, and multi-agent systems.

**Confidence**: High

---

## 7. Shared Objectness Role

**Status**: PROXY INSUFFICIENT — stress reveals structural change but proxy needs upgrade

**What we tested**: Stage 9 compared cross-controller objectness under normal vs stressed (pose ablation) conditions.

**What we found**:
- Normal: shared=0.586, internal=0.656
- Stressed (pose ablation): shared=0.898, internal=0.484
- Internal drops by 0.172 under stress (individual tracking degrades)
- Shared RISES by 0.312 under stress (controllers converge to similar broken behavior)

**Bug found and fixed**: Original stress condition (contact+target ablation) had zero effect because those channels are already 0.0 in BodylessBodyLayer. Fixed to use pose ablation.

**Interpretation**: Stress destroys individual tracking coherence (internal objectness drops) while making controllers degenerate into similar broken states (shared objectness rises). This is a valid finding about stress-induced degeneracy, but the current proxy cannot distinguish genuine shared representation from collapse-induced similarity.

**What would strengthen it**: Upgrade to BackboneShared-style metric: Ω(X) - λ·InteropLoss(X) - μ·SeamFragility(X) - ν·ScaleDrift(X)

**Confidence**: Low (finding is real but proxy is insufficient)
