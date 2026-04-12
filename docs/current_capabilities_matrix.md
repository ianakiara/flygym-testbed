# Current capabilities matrix

| Capability | Status | Notes |
| --- | --- | --- |
| Fly body construction | Implemented | Active in `compose/fly.py` |
| World composition | Implemented | `BaseWorld`, `FlatGroundWorld`, `TetheredWorld` |
| CPU simulation | Implemented | `simulation.py` |
| GPU batch simulation | Implemented | `warp/simulation.py` |
| Contact sensing | Implemented | Flat-ground leg contact sensors |
| Actuator force readout | Implemented | Exposed through `Simulation` APIs |
| Camera tracking | Implemented | `Fly.add_tracking_camera()` |
| Gymnasium interface | Legacy-only | Moved to `flygym-gymnasium` |
| High-level brain/body/world interfaces | Added in branch | `src/flygym_research/cognition/interfaces` |
| Reduced descending adapter | Added in branch | `src/flygym_research/cognition/adapters/descending_adapter.py` |
| Ascending summary adapter | Added in branch | `src/flygym_research/cognition/adapters/ascending_adapter.py` |
| Body-preserving cognition env | Added in branch | `FlyBodyWorldEnv` |
| Avatar/remapped world | Added in branch | `AvatarRemappedWorld`, `FlyAvatarEnv` |
| Benchmark harness | Added in branch | `experiments/benchmark_harness.py` |
| Baseline controllers | Added in branch | reflex, random, reduced, no-feedback, raw, bodyless |
| Falsification-first metrics | Added in branch | performance, stability, persistence, history, self/world, seam |
| Vision system package | Not present in active v2 code | Mentioned in docs, not in active packaged modules |
| Olfaction package | Not present in active v2 code | Mentioned in docs, not in active packaged modules |
| Rule-based locomotion controller | Not present in active v2 code | Mentioned in docs/citation summary only |
| RL navigation stack | Not present in active v2 code | Mentioned in docs/citation summary only |
