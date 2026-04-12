# Reduced descending interface

## Intent-level action fields

The milestone-1 reduced descending command is represented by `DescendingCommand` and currently includes:

- `move_intent`
- `turn_intent`
- `speed_modulation`
- `stabilization_priority`
- `approach_avoid`
- `interaction_trigger`
- `target_bias`
- `state_mode`

## Mapping strategy

Implemented in `src/flygym_research/cognition/adapters/descending_adapter.py`.

The adapter maps intent-level control into:
- position-actuator setpoints over active leg DoFs
- per-leg adhesion states
- logged seam metadata for later analysis

## Current milestone-1 behavior

- uses a tripod-style phase oscillator in the body layer
- modulates swing amplitude using move and speed intent
- modulates left/right bias using turn intent and target bias
- adds simple stabilization corrections from thorax orientation proxies
- falls back to neutral stance when high-level motion intent is near zero

## Important limitation

This is intentionally a research wrapper, not a claim that FlyGym already ships a validated high-level locomotion interface.
