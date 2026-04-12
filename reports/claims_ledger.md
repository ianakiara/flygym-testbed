# Claims Ledger

## Summary

- Total claims evaluated: 9
- Promoted: 3
- Useful: 2
- Weak / Not validated: 4

---

## PROMOTED Claims

### CLM-001: Ascending feedback loop is a structural prerequisite for coherent control

**Status**: PROMOTED  
**Experiment**: Stage 4 — Ascending/descending loop validation  
**Evidence**:
- Pose channel ablation causes 100% stability drop (1.0 → 0.0)
- All-channels-off ablation causes identical 100% stability drop
- Locomotion, contact, target, internal ablations cause 0% stability drop individually
- Different channels produce different collapse signatures (STRONG PASS)
- 10 seeds, consistent across all seeds (std ≈ 0)

**Ablations survived**: pose ablation, all-off ablation, per-channel ablation sweep  
**Negative controls**: locomotion/contact/target/internal ablation correctly show no stability effect  
**Failure modes**: Pose channel is the critical stability channel; others are structurally redundant for stability  
**Strongest falsifier**: If pose ablation had no effect, the loop would be structurally irrelevant  
**Next experiment**: Test with embodied FlyBodyLayer (MuJoCo) to confirm pose channel importance with real physics  
**Risk if wrong**: Architecture overfits to a single channel dependency rather than true feedback loop integration  
**Caveats**:
- Tested only with BodylessBodyLayer (stability is a synthetic metric in this mode)
- Pose group contains {stability, thorax_height_mm, body_speed_mm_s} — the "stability" feature directly feeds the metric

---

### CLM-002: Controller interoperability produces translatable shared structure

**Status**: PROMOTED  
**Experiment**: Stage 7 — Controller interoperability validation  
**Evidence**:
- Translated alignment mean: 0.437
- Raw alignment mean: 0.000
- Gap: 0.437 (threshold was 0.10)
- Tested across 10 controller pairs (5 controllers)
- Highest interop: reflex_only vs raw_control (0.500), reduced_descending vs memory (0.499)
- Lowest interop: reflex_only vs memory (0.389)

**Ablations survived**: Different controller families (rule-based, memory-augmented, planner, raw)  
**Negative controls**: Raw correlation is 0.000 — controllers share NO surface-level latent structure  
**Failure modes**: Interop score is driven by reward trajectory similarity; controllers that solve the same task share structure  
**Strongest falsifier**: If random vs. structured controllers showed equal interop, the finding would be trivial  
**Next experiment**: Test with learned representations (RL controller) to see if interop holds with trained policies  
**Risk if wrong**: Interop score may measure task similarity rather than true representational alignment  
**Caveats**:
- All controllers tested in same world (avatar_remapped)
- Interop metric combines reward correlation and action distribution similarity
- No cross-world interop tested yet

---

### CLM-003: Seam perturbation predicts system failure better than local module quality

**Status**: PROMOTED  
**Experiment**: Stage 8 — Seam / composition validation  
**Evidence**:
- Breaking pose+locomotion channels: stability drops 100% (1.0 → 0.0), but reward IMPROVES by 26.05
- All-channels-broken: identical pattern — stability collapse, reward improvement
- Sensor dropout (contact+target): no effect on stability or reward
- Local metric (reward) says "system improved" while seam metric (stability) says "system collapsed"

**Ablations survived**: broken_descending, all_channels_broken  
**Negative controls**: sensor_dropout correctly shows no system-level failure  
**Failure modes**: Seam metric (stability) catches structural damage that reward misses  
**Strongest falsifier**: If broken seams always caused proportional reward drops, local metrics would suffice  
**Next experiment**: Add latency injection, compression loss, and schema mismatch perturbations  
**Risk if wrong**: Stability metric may be too tightly coupled to pose channel, making this a circular finding  
**Caveats**:
- Reward improvement under ablation suggests the baseline controller may over-prioritize stability
- Only tested with BodylessBodyLayer
- Seam fragility metric itself showed 0.000 delta — the finding comes from stability, not the seam_fragility metric

---

## USEFUL Claims

### CLM-004: Reduced descending architecture provides meaningful performance improvement

**Status**: USEFUL  
**Experiment**: Stage 2 — Baseline truth validation  
**Evidence**:
- Reduced descending mean return: +131.19 over reflex-only across 3 worlds
- Native world: reduced_descending 214.4 vs reflex_only 6.4 (+208.0)
- Simplified world: reduced_descending -68.6 vs reflex_only -284.3 (+215.7)
- Avatar world: reduced_descending -118.2 vs reflex_only -88.1 (-30.1, worse)

**Blockers**: Avatar world shows reduced_descending performing WORSE than reflex_only  
**Strongest falsifier**: Avatar world result (reduced descending is worse)  
**Next experiment**: Investigate why reduced descending underperforms in avatar world  
**Risk if wrong**: Architecture advantage may be world-specific, not general

---

### CLM-005: Stability and persistence transfer across world types

**Status**: USEFUL  
**Experiment**: Stage 10 — Transfer validation  
**Evidence**:
- Stability: nonzero in 3/3 worlds (simplified, avatar, native)
- Persistence (state_autocorrelation): nonzero in 3/3 worlds
- History dependence: nonzero in only 1/3 worlds (fails transfer)
- Self/world marker: nonzero in only 1/3 worlds (fails transfer)

**Blockers**: Only 2 of 4 findings transfer  
**Next experiment**: Test transfer under controller swap and sensor degradation  

---

## WEAK / NOT VALIDATED Claims

### CLM-006: Body substrate improves structural control metrics

**Status**: NOT VALIDATED (infrastructure limitation)  
**Experiment**: Stage 3 — Body substrate value validation  
**Evidence**: Body wins 0/4 structure metrics across all worlds (need ≥3/4 in ≥2 worlds)  
**Root cause**: All environments use BodylessBodyLayer because MuJoCo/EGL is not available on the test machine. The comparison is bodyless vs bodyless — structurally meaningless.  
**Next experiment**: Run on machine with MuJoCo/EGL support using FlyBodyLayer  
**Risk if wrong**: The entire body substrate hypothesis is untested

---

### CLM-007: History dependence is structurally real

**Status**: ARTIFACT  
**Experiment**: Stage 5 — History dependence / hysteresis validation  
**Evidence**: Random controller (0.291) shows MORE history dependence than memory controller (0.008)  
**Root cause**: The metric (action variance in coarse-state buckets) measures noise, not meaningful path dependence. Random actions maximize variance by definition.  
**Next experiment**: Replace with prediction-based metric (next-regime AUC, repair cost R²)  
**Risk if wrong**: History dependence claim is based on a flawed metric

---

### CLM-008: Self/world disambiguation is a candidate prerequisite

**Status**: WEAK  
**Experiment**: Stage 6 — Self/world disambiguation validation  
**Evidence**: Reduced descending |self_world_marker| = 0.0291; random baseline = 0.0292. No separation.  
**Root cause**: External events in AvatarRemappedWorld are too subtle (noise_scale=0.02, period=7 steps). Correlation-based metric cannot detect the signal.  
**Next experiment**: Amplify perturbations, use intervention-based rather than correlation-based measurement  

---

### CLM-009: Shared objectness beats internal objectness under stress

**Status**: WEAK  
**Experiment**: Stage 9 — Shared objectness validation  
**Evidence**: Shared objectness mean 0.586 vs internal 0.585. Gap = 0.001 (threshold: 0.05).  
**Root cause**: Current objectness proxy (target distance CV + persistence) is too simplistic to capture cross-controller object coherence.  
**Next experiment**: Use learned latent representations and proper object tracking across controller swaps  
