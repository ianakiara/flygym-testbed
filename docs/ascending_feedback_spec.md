# Ascending feedback specification

## Two-tier structure

### Raw tier

The body layer exposes raw feedback directly from FlyGym simulation APIs:

- joint angles
- joint velocities
- body positions
- body rotations
- contact flags
- contact forces and torques
- contact positions, normals, and tangents
- actuator forces

### Summary tier

The milestone-1 ascending adapter computes compact summary features such as:

- stability
- thorax height
- body speed
- contact fraction
- slip risk
- collision load
- locomotion quality
- actuator effort
- target distance and target salience
- internal phase and phase velocity

## Ablations

Feedback-channel ablations are supported through `BodyLayerConfig.disabled_feedback_channels`.

Current channel groups:
- `pose`
- `contact`
- `locomotion`
- `target`
- `internal`

### Ablation behavior

When a channel group is disabled:
- the corresponding feature keys are **zeroed** (set to `0.0`), not removed from the dict
- the observation shape remains consistent across ablation conditions
- the `active_channels` tuple excludes zeroed keys
- the `disabled_channels` tuple records which groups were ablated
