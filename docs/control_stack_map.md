# Control stack map

## Current active stack in the repo

1. **Anatomy and pose presets**
   - segment naming
   - joint presets
   - actuated-DoF presets
   - neutral pose presets
2. **Fly composition**
   - body tree
   - joints
   - actuators
   - adhesion
   - cameras
3. **World composition**
   - free or tethered attachment
   - ground contacts
   - environmental MJCF root
4. **Simulation**
   - compile MJCF to MuJoCo
   - step physics
   - expose state and control APIs
5. **Rendering / Warp**
   - viewer and video output
   - GPU batched stepping

## Milestone-1 cognition stack added in this branch

1. **Brain layer**
   - baseline controllers
   - persistent controller state hooks
2. **Body/reflex layer**
   - `FlyBodyLayer`
   - reduced descending mapping
   - actuator and adhesion control
   - ascending summaries
3. **World layer**
   - native physical task wrapper
   - simplified embodied task wrapper
   - avatar/remapped world
4. **Environment wrapper**
   - `FlyBodyWorldEnv`
   - `FlyAvatarEnv`
   - `FlyDualWorldEnv`
5. **Benchmark layer**
   - episode runner
   - baseline suite
   - falsification-first metrics
