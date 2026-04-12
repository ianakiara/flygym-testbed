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

**Status**: CONFIRMED — strongest finding

**What we tested**: Stage 7 computed translated vs. raw alignment across 10 controller pairs in the same world.

**What we found**:
- Mean translated alignment: 0.437
- Mean raw alignment: 0.000
- Gap: 0.437 (threshold was 0.10)
- Highest pair: reflex_only vs raw_control (0.500)
- Lowest pair: memory vs reflex_only (0.389)

**Interpretation**: Controllers that use completely different internal mechanisms (reflexive, memory-augmented, planning-based, raw control) nonetheless converge on similar task solutions. The convergence is in outcome/reward space, not in latent state space (raw correlation = 0.000). This means there is genuine task structure that any adequate controller must respect.

**Promoted wording**: Translated controller alignment reveals shared structure that raw alignment misses.

**Caveats**:
- Tested only in avatar_remapped world
- No RL controller tested
- Interop metric includes reward trajectory similarity, which may dominate

**Confidence**: High

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
