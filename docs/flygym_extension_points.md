# FlyGym extension points

## Recommended extension seams

### 1. Fly construction and actuator setup

Use `Fly`, `Skeleton`, `KinematicPosePreset`, and `add_actuators()` to build reusable body substrates.

Why this matters:
- lets the research layer preserve embodiment
- avoids patching MuJoCo internals directly
- keeps reduced descending control as an adapter over existing actuators

### 2. World composition

Use `BaseWorld` subclasses as the boundary for physical-world changes.

Relevant seams:
- `BaseWorld.add_fly()`
- concrete `_attach_fly_mjcf()` implementations
- ground-contact sensor setup in `FlatGroundWorld`

### 3. Simulation stepping and state access

Use `Simulation` and `GPUSimulation` as the stepping boundary.

Relevant read APIs:
- joint angles
- joint velocities
- body positions
- body rotations
- actuator forces
- ground-contact info

Relevant write APIs:
- `set_actuator_inputs()`
- `set_leg_adhesion_states()`

### 4. Pose and anatomy presets

Use anatomy, joint, and pose presets as the stable control-surface for reduced action mappings.

### 5. Logging and wrapper layer

Wrap core APIs in a research package rather than adding cognition-specific logic directly into `src/flygym`.

## Branch implementation choice

This branch follows that guidance by adding `src/flygym_research/cognition` as an outer wrapper layer around:

- `Fly`
- `FlatGroundWorld`
- `Simulation`
- state read/write APIs
