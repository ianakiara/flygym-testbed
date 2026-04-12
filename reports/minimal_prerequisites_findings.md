# Minimal Prerequisites Findings

## Summary

This report summarizes what the 10-stage validation program found about each candidate prerequisite for persistent integrated control. All findings are framed as engineering/behavioral properties, not consciousness claims.

---

## 1. Body Role

**Status**: NOT VALIDATED (infrastructure limitation)

**What we tested**: Stage 3 compared body-preserving controllers (reduced_descending, raw_control) against bodyless controllers across simplified and avatar worlds.

**What we found**: Body-preserving agents won 0/4 structure metrics in 3 out of 4 world/controller combinations. One combination (avatar/raw_control) won 1/4.

**Why this doesn't answer the question**: All environments used BodylessBodyLayer (no MuJoCo/EGL available). The "body-preserving" and "bodyless" agents are using the same body layer — the comparison is meaningless.

**What would answer it**: Run the same tests with FlyBodyLayer (requires MuJoCo simulation) on a machine with EGL/GPU support. Compare BodylessBodyLayer vs FlyBodyLayer directly.

**Confidence**: N/A — untested

---

## 2. Stabilization Role

**Status**: CONFIRMED as structurally critical

**What we tested**: Stage 4 ablated individual ascending feedback channels and measured stability collapse.

**What we found**:
- Pose channel ablation → 100% stability drop (1.0 → 0.0)
- All-channels-off → 100% stability drop
- Individual other channels (locomotion, contact, target, internal) → 0% stability drop
- 10 seeds, zero variance in these results

**Interpretation**: Stabilization depends critically on pose-group feedback (stability, thorax_height, body_speed). Without it, the system loses all stability even if task performance changes. This is a structural prerequisite.

**Caveat**: The stability metric is computed FROM pose features, so zeroing pose and measuring stability is partially tautological. The strong version of this claim needs behavioral validation (does the agent fall over, fail tasks, produce erratic trajectories?).

**Confidence**: Medium-High

---

## 3. Self/World Role

**Status**: WEAK — no clear signal

**What we tested**: Stage 6 compared self_world_separation scores across controllers (reduced_descending, memory, reflex_only, bodyless, random).

**What we found**:
- Reduced descending |self_world_marker|: 0.0291
- Random baseline |self_world_marker|: 0.0292
- Gap: -0.0001 (random is marginally better)

**Why it failed**:
1. The external events in AvatarRemappedWorld fire every 7 steps with noise_scale=0.02 — too subtle
2. The correlation-based metric (speed vs. event correlation) doesn't capture the distinction
3. All controllers produce nearly identical self/world scores

**What would work better**:
- Stronger perturbations (larger noise, more frequent events)
- Intervention-based measurement (inject self-motion, measure compensation)
- Separate self-motion and world-motion perturbation experiments

**Confidence**: Low

---

## 4. History Role

**Status**: ARTIFACT — current metric is flawed

**What we tested**: Stage 5 compared history dependence scores across reduced_descending, memory, and random controllers.

**What we found**:
- Random: 0.291 (highest "history dependence")
- Reduced descending: 0.010
- Memory: 0.008

**Why this is wrong**: The metric measures action variance within coarse-state buckets. Random actions have maximum variance by definition, so random scores highest. This measures noise, not meaningful path dependence.

**What would work**: Prediction-based metrics:
- Train state-only and history-aware predictors on action sequences
- Compare next-action prediction AUC
- Compare next-regime prediction AUC
- Compare repair cost prediction R²
- If history-aware predictor significantly outperforms state-only, history dependence is real

**Confidence**: High that the current finding is an artifact; N/A for the actual question

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

**Caveats**:
- The seam_fragility metric itself showed 0.000 delta — the finding comes from stability drop, not the seam metric
- Reward improvement may indicate the baseline controller over-weights stability
- Only tested with BodylessBodyLayer

**Transferable lesson**: In any modular system (not just this one), broken interfaces between modules can produce locally-improved metrics while destroying global coherence. This applies to robotics, distributed systems, microservices, and multi-agent systems.

**Confidence**: High
