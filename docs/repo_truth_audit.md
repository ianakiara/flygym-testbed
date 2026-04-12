# Repo truth audit

## Executive summary

This fork is currently a high-quality FlyGym 2.x simulation substrate, not yet a cognition benchmarking framework.

## Active code path

- Active implementation: `/home/runner/work/flygym-testbed/flygym-testbed/src/flygym`
- Core active modules:
  - `compose/` for fly and world composition
  - `simulation.py` for CPU stepping
  - `warp/` for GPU batch stepping
  - `rendering.py` for viewer and video output
- Tests actively exercised in CI:
  - `/home/runner/work/flygym-testbed/flygym-testbed/tests/core`
  - `/home/runner/work/flygym-testbed/flygym-testbed/tests/examples`

## Legacy and archived carryovers

- FlyGym 1.x / Gymnasium is no longer the active interface.
- The repo explicitly points users to `flygym-gymnasium` for the older API.
- Remaining in-repo legacy carryovers are limited to:
  - migration docs
  - `flygym.utils.api1to2`
  - legacy MJCF/config assets under `src/flygym/assets/model/legacy`

## What is implemented today

### Research-quality substrate pieces

- Fly morphology, rigging, pose presets, and contact presets
- World composition via `BaseWorld`, `FlatGroundWorld`, and `TetheredWorld`
- CPU simulation with direct state read/write APIs
- GPU batch simulation with Warp
- Ground-contact sensing and actuator-force access
- Tracking camera support
- Motion-snippet replay demo data and benchmark demo code

### Prototype-quality or absent pieces

- No packaged higher-level cognition architecture
- No explicit brain/body/world interface package
- No reduced descending controller abstraction in the active code path
- No ascending summary adapter in the active code path
- No custom cognition environments or world-remapping layer
- No benchmark harness for falsification-first cognition studies
- No claims ledger or anti-overclaim report outputs

## Repo split assessment

- Core simulation and composition code is actively maintained and tested.
- Documentation still mentions paper-era capabilities such as vision, olfaction, rule-based control, hybrid control, and RL navigation, but those pieces are not present as active packaged modules in this fork.
- This mismatch is the main truth-audit finding that must stay visible in downstream reports.

## Milestone-1 additions in this branch

This branch adds a separate research package under `src/flygym_research/cognition` that wraps FlyGym rather than patching core internals. It introduces:

- explicit interfaces for brain/body/world layers
- reduced descending and ascending adapter modules
- body-preserving and avatar-remapped environments
- initial baseline controllers
- falsification-first metric functions
- a lightweight benchmark harness
