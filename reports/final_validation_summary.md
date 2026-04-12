# Final Validation Summary

## Overall Result: 6/10 stages passed

| Stage | Result | Key Finding |
|-------|--------|-------------|
| 1. Sanity | **PASS** | All 8 controllers run in all worlds, deterministic seeding confirmed, no unexpected NaNs, all ablations config-toggleable |
| 2. Baseline Truth | **PASS** | Reduced descending beats reflex-only by +131.19 mean return across worlds |
| 3. Body Substrate | **FAIL** | Cannot validate — all envs use BodylessBodyLayer (no MuJoCo/EGL on test machine). Body wins 0/4 structure metrics because comparison is bodyless vs bodyless. |
| 4. Ascending Loop | **PASS (STRONG)** | Pose channel ablation causes 100% stability drop. Multiple channels produce different collapse signatures. |
| 5. History Dependence | **FAIL** | Random controller shows MORE history dependence (0.291) than memory (0.008) — metric measures action variance in state buckets, which random maximizes by definition. Metric design flaw. |
| 6. Self/World | **FAIL** | Reduced descending self_world_marker (0.0291) does not beat random baseline (0.0292). Signal too weak to distinguish. |
| 7. Interoperability | **PASS** | Translated alignment (0.437) massively beats raw alignment (0.000). Gap of 0.437 far exceeds 0.10 threshold. |
| 8. Seam Stress | **PASS** | Breaking descending channels (pose+locomotion) causes 100% stability drop even when raw performance improves (-26.05). Seam perturbation predicts structural failure. |
| 9. Shared Objectness | **FAIL** | Shared objectness (0.586) barely beats internal (0.585). Gap of 0.001 far below 0.05 threshold. |
| 10. Transfer | **PASS** | Stability and persistence survive across all 3 worlds. History and self/world do not transfer. |

---

## What Passed

1. **Infrastructure is sound** — deterministic, no silent NaNs, all controller/world combinations work, ablations toggle by config only.
2. **Reduced descending architecture adds value** — +131.19 return over reflex-only across worlds. Not trivial.
3. **Ascending feedback loop matters structurally** — pose channel ablation destroys stability (1.0 → 0.0) without total task failure. Different channels produce different collapse signatures. This is a **strong** finding.
4. **Controller interoperability is real** — translated alignment massively outperforms raw alignment (0.437 vs 0.000). Controllers share task structure even when low-level behavior differs.
5. **Seam perturbation predicts failure** — breaking pose+locomotion channels causes stability collapse even when reward improves. Local metrics mislead; seam metrics catch it. This is a **strong** finding.
6. **Stability and persistence transfer across worlds** — these findings are not local hacks.

## What Failed

1. **Body substrate value** — CANNOT BE VALIDATED without MuJoCo physics simulation. All tests use BodylessBodyLayer which trivially returns stability=1.0. This is an infrastructure limitation, not a disproof.
2. **History dependence metric** — the metric (action variance in coarse-state buckets) is flawed. Random actions maximize this metric by definition. Need a better metric: prediction-based (does knowing history improve next-action prediction?) rather than variance-based.
3. **Self/world disambiguation** — signal is too weak (0.029 vs 0.029). The external event mechanism in AvatarRemappedWorld only fires every 7 steps with tiny noise (0.02 scale), which is too subtle for correlation-based detection.
4. **Shared objectness** — gap is negligible (0.001). The current objectness proxy (target distance CV + persistence) doesn't capture true cross-controller objectness.

## What Is Hype

- **History dependence as currently measured** — the variance-based metric is an artifact. It measures noise, not meaningful path dependence.
- **Self/world separation** — the correlation-based metric doesn't produce meaningful signal in the current environment setup.
- **Shared objectness** — the proxy score doesn't differentiate meaningfully from plain internal objectness.

## What Deserves Promotion

1. **Ascending feedback loop structure** → PROMOTED (strong evidence)
   - Pose channel removal causes 100% stability collapse
   - Different channels produce different collapse patterns
   - Not explained by trivial performance drop
   - Survives across configurations

2. **Controller interoperability** → PROMOTED (strong evidence)
   - 0.437 translated vs 0.000 raw alignment
   - Consistent across all 10 controller pairs
   - Shows shared task structure independent of behavioral details

3. **Seam fragility predicts system failure** → PROMOTED (strong evidence)
   - Broken seams cause stability collapse even when reward improves
   - Local metrics (reward) are misleading; seam metrics catch structural damage
   - This is the most transferable finding

## What Needs Sharper Falsification

1. **Body substrate value** — needs MuJoCo/EGL environment to test properly
2. **History dependence** — needs prediction-based metric (next-regime AUC, repair cost R²)
3. **Self/world separation** — needs stronger perturbation protocol and better metric
4. **Shared objectness** — needs proper cross-controller stress tests with learned representations

---

## Evidence Tier Assessment

| Finding | Tier | Justification |
|---------|------|---------------|
| Ascending loop matters | **Strong → Promoted** | Survives ablation, different collapse signatures, not trivial |
| Controller interoperability | **Strong → Promoted** | 0.437 gap, consistent across pairs, transfers |
| Seam fragility law | **Strong → Promoted** | Local metrics mislead, seam metrics predict failure |
| Stability transfers | **Useful** | Survives 3 worlds but not stress-tested under ablation |
| State persistence transfers | **Useful** | Survives 3 worlds but autocorrelation is a weak proxy |
| Body substrate value | **Not validated** | Infrastructure limitation |
| History dependence | **Artifact** | Metric is flawed |
| Self/world separation | **Weak** | Signal too small to distinguish from noise |
| Shared objectness | **Weak** | Gap negligible |

---

## Forbidden Words Check

This report contains NONE of: consciousness, sentience, phenomenology, "the fly is aware."

All findings are framed as: candidate prerequisites, persistent integrated control markers, structural organization properties, controller-invariant representations, seam fragility laws.
