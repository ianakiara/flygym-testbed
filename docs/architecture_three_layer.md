# Three-layer architecture

## Design goal

Preserve the body substrate while making higher-level control explicit and world-swappable.

## Layers

```text
Brain layer
  ↓ reduced descending command
Body/reflex layer
  ↓ body-world interaction + raw state
World layer
  ↓ task state, reward, delayed consequences
Brain layer receives ascending summary + raw/body state access
```

## Concrete branch implementation

### Brain layer

Implemented in `src/flygym_research/cognition/controllers`.

Current milestone-1 brain interfaces support:
- reflex-only control
- reduced descending control
- no-ascending-feedback ablation
- random control
- raw low-level control baseline
- bodyless avatar baseline controller

### Body/reflex layer

Implemented in `src/flygym_research/cognition/body_reflex.py`.

Responsibilities:
- build the fly body and actuator substrate
- step the MuJoCo simulation
- convert reduced descending commands into actuator and adhesion commands
- expose raw feedback and summarized ascending feedback

### World layer

Implemented in `src/flygym_research/cognition/worlds.py`.

Modes:
- native physical world task wrapper
- simplified embodied task wrapper
- avatar/remapped world

## Why this architecture is safe for the repo

- core FlyGym internals are wrapped, not rewritten
- research logic lives in a separate package
- world swapping happens above the body layer
- baseline and ablation logic stays outside `src/flygym`
