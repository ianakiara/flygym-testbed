# Weights and configs inventory

## Trained weights and checkpoints actually present

No pretrained controller checkpoints were found in the active code path.

### Present non-mesh assets

- `src/flygym_demo/spotlight_data/assets/spotlight_behavior_clip.npz`
  - bundled motion-snippet demo data
  - used for replay and benchmarking demos
  - not a learned controller checkpoint

## Config assets actually present

### Active v2 config assets

- `src/flygym/assets/model/rigging.yaml`
- `src/flygym/assets/model/mujoco_globals.yaml`
- `src/flygym/assets/model/visuals.yaml`
- `src/flygym/assets/model/pose/neutral/*.yaml`

### Legacy carryovers

- `src/flygym/assets/model/legacy/flygym1_config.yaml`
- `src/flygym/assets/model/legacy/*.xml`

## New cognition configs added in this branch

Code-level dataclass configs were added for milestone 1 under:

- `src/flygym_research/cognition/config.py`

These configs currently cover:

- body/reflex timing and gait adapter parameters
- feedback-channel ablations
- environment episode length
- avatar remapping scales and noise
