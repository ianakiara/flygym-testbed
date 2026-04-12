# World modes specification

## Mode A — Native physical world

- preserves the FlyGym body and physical stepping
- task proxy rewards forward progress and stability
- serves as the closest control condition to stock FlyGym stepping

## Mode B — Simplified embodied world

- preserves the same physical body substrate
- adds a simple target-approach task over the physical body state
- exposes delayed consequences through repeated stepping and target distance

## Mode C — Avatar/remapped world

- keeps the body/reflex layer running
- remaps higher-level behavior into an abstract avatar state
- keeps noise, delayed consequences, and external perturbation events
- supports bodyless comparison through `BodylessBodyLayer`

## Cross-mode comparability

All three modes emit:
- world mode identifier
- reward
- target vector or equivalent task state
- step count
- info payload for external-event logging
