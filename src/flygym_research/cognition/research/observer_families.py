"""Observer families for PR #5 — 9 perturbation types.

Extends the existing 4 types (noise, scaling, bias, partial) with:
  - dimension masking
  - dimension permutation
  - sign flips
  - nonlinear warp
  - mixed-family distortion
"""

from __future__ import annotations

import numpy as np

from ..interfaces import (
    AscendingSummary,
    BrainObservation,
    StepTransition,
    WorldState,
)


# ---------------------------------------------------------------------------
# Helper: deep-clone a transition with perturbed state
# ---------------------------------------------------------------------------

def _clone_transition(
    t: StepTransition,
    *,
    body_positions: np.ndarray | None = None,
    features: dict[str, float] | None = None,
    observables: dict | None = None,
) -> StepTransition:
    """Clone a transition with optional overrides for body, features,
    and world observables."""
    obs = t.observation
    raw = obs.raw_body

    if body_positions is not None:
        raw = type(raw)(
            time=raw.time,
            joint_angles=raw.joint_angles,
            joint_velocities=raw.joint_velocities,
            body_positions=body_positions,
            body_rotations=raw.body_rotations,
            contact_active=raw.contact_active,
            contact_forces=raw.contact_forces,
            contact_torques=raw.contact_torques,
            contact_positions=raw.contact_positions,
            contact_normals=raw.contact_normals,
            contact_tangents=raw.contact_tangents,
            actuator_forces=raw.actuator_forces,
        )

    summary = obs.summary
    if features is not None:
        summary = AscendingSummary(
            features=features,
            active_channels=summary.active_channels,
            disabled_channels=summary.disabled_channels,
        )

    world = obs.world
    if observables is not None:
        world = WorldState(
            mode=world.mode,
            step_count=world.step_count,
            reward=world.reward,
            terminated=world.terminated,
            truncated=world.truncated,
            observables=observables,
            info=world.info,
        )

    new_obs = BrainObservation(
        raw_body=raw,
        summary=summary,
        world=world,
        history=obs.history,
    )
    return StepTransition(
        observation=new_obs,
        action=t.action,
        reward=t.reward,
        terminated=t.terminated,
        truncated=t.truncated,
        info=t.info,
    )


def _copy_world_observables(
    obs_dict: dict,
    body_positions: np.ndarray,
) -> dict:
    """Update world observables to reflect perturbed body positions."""
    result = dict(obs_dict)
    if body_positions.shape[0] > 0:
        result["avatar_xy"] = body_positions[0, :2].copy()
    return result


# ---------------------------------------------------------------------------
# Perturbation families
# ---------------------------------------------------------------------------

def perturb_noise(
    transitions: list[StepTransition],
    *,
    scale: float = 0.5,
    rng: np.random.Generator | None = None,
) -> list[StepTransition]:
    """Additive Gaussian noise to body positions, features, observables."""
    rng = rng or np.random.default_rng(42)
    result = []
    for t in transitions:
        pos = t.observation.raw_body.body_positions.copy()
        pos += rng.normal(0, scale, size=pos.shape).astype(pos.dtype)

        feats = dict(t.observation.summary.features)
        for k in feats:
            feats[k] = float(feats[k]) + rng.normal(0, scale * 0.5)

        obs_dict = _copy_world_observables(
            dict(t.observation.world.observables), pos,
        )
        result.append(_clone_transition(
            t, body_positions=pos, features=feats, observables=obs_dict,
        ))
    return result


def perturb_scaling(
    transitions: list[StepTransition],
    *,
    factor: float = 2.5,
    rng: np.random.Generator | None = None,
) -> list[StepTransition]:
    """Multiplicative scaling of body positions, features, observables."""
    result = []
    for t in transitions:
        pos = t.observation.raw_body.body_positions.copy() * factor

        feats = dict(t.observation.summary.features)
        for k in feats:
            feats[k] = float(feats[k]) * factor

        obs_dict = _copy_world_observables(
            dict(t.observation.world.observables), pos,
        )
        result.append(_clone_transition(
            t, body_positions=pos, features=feats, observables=obs_dict,
        ))
    return result


def perturb_bias(
    transitions: list[StepTransition],
    *,
    bias: float = 2.0,
    rng: np.random.Generator | None = None,
) -> list[StepTransition]:
    """Fixed bias addition to body positions, features, observables."""
    result = []
    for t in transitions:
        pos = t.observation.raw_body.body_positions.copy() + bias

        feats = dict(t.observation.summary.features)
        for k in feats:
            feats[k] = float(feats[k]) + bias

        obs_dict = _copy_world_observables(
            dict(t.observation.world.observables), pos,
        )
        result.append(_clone_transition(
            t, body_positions=pos, features=feats, observables=obs_dict,
        ))
    return result


def perturb_partial(
    transitions: list[StepTransition],
    *,
    drop_fraction: float = 0.4,
    rng: np.random.Generator | None = None,
) -> list[StepTransition]:
    """Drop channels (partial observability) from body positions."""
    rng = rng or np.random.default_rng(42)
    result = []
    drop_idx = None
    feat_drop_keys: set[str] | None = None
    for t in transitions:
        pos = t.observation.raw_body.body_positions.copy()
        if drop_idx is None:
            n_bodies = pos.shape[0]
            n_drop = max(1, int(n_bodies * drop_fraction))
            drop_idx = rng.choice(n_bodies, size=n_drop, replace=False)
        pos[drop_idx] = 0.0

        feats = dict(t.observation.summary.features)
        feat_keys = sorted(feats.keys())
        if feat_drop_keys is None:
            n_feat_drop = max(1, int(len(feat_keys) * drop_fraction))
            feat_drop_keys = set(rng.choice(feat_keys, size=min(n_feat_drop, len(feat_keys)), replace=False))
        for k in feat_keys:
            if k in feat_drop_keys:
                feats[k] = 0.0

        obs_dict = _copy_world_observables(
            dict(t.observation.world.observables), pos,
        )
        result.append(_clone_transition(
            t, body_positions=pos, features=feats, observables=obs_dict,
        ))
    return result


def perturb_masking(
    transitions: list[StepTransition],
    *,
    mask_fraction: float = 0.5,
    rng: np.random.Generator | None = None,
) -> list[StepTransition]:
    """Zero out entire dimensions of body positions."""
    rng = rng or np.random.default_rng(42)
    result = []
    mask_dims = None
    feat_mask_keys: set[str] | None = None
    for t in transitions:
        pos = t.observation.raw_body.body_positions.copy()
        if mask_dims is None:
            n_dims = pos.shape[1] if pos.ndim > 1 else 1
            n_mask = max(1, int(n_dims * mask_fraction))
            mask_dims = rng.choice(n_dims, size=n_mask, replace=False)
        if pos.ndim > 1:
            pos[:, mask_dims] = 0.0

        feats = dict(t.observation.summary.features)
        feat_keys = sorted(feats.keys())
        if feat_mask_keys is None:
            n_feat_mask = max(1, int(len(feat_keys) * mask_fraction))
            feat_mask_keys = set(rng.choice(feat_keys, size=min(n_feat_mask, len(feat_keys)), replace=False))
        for k in feat_keys:
            if k in feat_mask_keys:
                feats[k] = 0.0

        obs_dict = _copy_world_observables(
            dict(t.observation.world.observables), pos,
        )
        result.append(_clone_transition(
            t, body_positions=pos, features=feats, observables=obs_dict,
        ))
    return result


def perturb_permutation(
    transitions: list[StepTransition],
    *,
    rng: np.random.Generator | None = None,
) -> list[StepTransition]:
    """Permute body indices — shuffle which body part is which."""
    rng = rng or np.random.default_rng(42)
    result = []
    perm = None
    for t in transitions:
        pos = t.observation.raw_body.body_positions.copy()
        if perm is None:
            perm = rng.permutation(pos.shape[0])
        pos = pos[perm]

        feats = dict(t.observation.summary.features)
        feat_keys = sorted(feats.keys())
        feat_vals = [feats[k] for k in feat_keys]
        if feat_perm is None:
            feat_perm = rng.permutation(len(feat_vals))
        feats = dict(zip(feat_keys, [feat_vals[i] for i in feat_perm]))

        obs_dict = _copy_world_observables(
            dict(t.observation.world.observables), pos,
        )
        result.append(_clone_transition(
            t, body_positions=pos, features=feats, observables=obs_dict,
        ))
    return result


def perturb_sign_flip(
    transitions: list[StepTransition],
    *,
    flip_fraction: float = 0.5,
    rng: np.random.Generator | None = None,
) -> list[StepTransition]:
    """Flip the sign of random dimensions."""
    rng = rng or np.random.default_rng(42)
    result = []
    flip_mask = None
    for t in transitions:
        pos = t.observation.raw_body.body_positions.copy()
        if flip_mask is None:
            n_total = pos.size
            n_flip = max(1, int(n_total * flip_fraction))
            flat_idx = rng.choice(n_total, size=n_flip, replace=False)
            flip_mask = np.ones(n_total, dtype=np.float64)
            flip_mask[flat_idx] = -1.0
            flip_mask = flip_mask.reshape(pos.shape)
        pos = pos * flip_mask

        feats = dict(t.observation.summary.features)
        feat_keys = sorted(feats.keys())
        if feat_flip_keys is None:
            n_feat_flip = max(1, int(len(feat_keys) * flip_fraction))
            feat_flip_keys = set(rng.choice(feat_keys, size=min(n_feat_flip, len(feat_keys)), replace=False))
        for k in feat_keys:
            if k in feat_flip_keys:
                feats[k] = -float(feats[k])

        obs_dict = _copy_world_observables(
            dict(t.observation.world.observables), pos,
        )
        result.append(_clone_transition(
            t, body_positions=pos, features=feats, observables=obs_dict,
        ))
    return result


def perturb_nonlinear(
    transitions: list[StepTransition],
    *,
    power: float = 2.0,
    rng: np.random.Generator | None = None,
) -> list[StepTransition]:
    """Nonlinear warp: x → sign(x) * |x|^power."""
    result = []
    for t in transitions:
        pos = t.observation.raw_body.body_positions.copy()
        pos = np.sign(pos) * np.abs(pos) ** power

        feats = dict(t.observation.summary.features)
        for k in feats:
            v = float(feats[k])
            feats[k] = float(np.sign(v) * abs(v) ** power)

        obs_dict = _copy_world_observables(
            dict(t.observation.world.observables), pos,
        )
        result.append(_clone_transition(
            t, body_positions=pos, features=feats, observables=obs_dict,
        ))
    return result


def perturb_mixed(
    transitions: list[StepTransition],
    *,
    rng: np.random.Generator | None = None,
) -> list[StepTransition]:
    """Mixed-family distortion: applies noise + scaling + sign flip."""
    rng = rng or np.random.default_rng(42)
    out = perturb_noise(transitions, scale=0.3, rng=rng)
    out = perturb_scaling(out, factor=1.5, rng=rng)
    out = perturb_sign_flip(out, flip_fraction=0.3, rng=rng)
    return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

OBSERVER_FAMILIES: dict[str, object] = {
    "noise": perturb_noise,
    "scaling": perturb_scaling,
    "bias": perturb_bias,
    "partial": perturb_partial,
    "masking": perturb_masking,
    "permutation": perturb_permutation,
    "sign_flip": perturb_sign_flip,
    "nonlinear": perturb_nonlinear,
    "mixed": perturb_mixed,
}


def apply_observer_perturbation(
    transitions: list[StepTransition],
    family_name: str,
    *,
    rng: np.random.Generator | None = None,
    **kwargs: object,
) -> list[StepTransition]:
    """Apply a named observer perturbation family."""
    fn = OBSERVER_FAMILIES.get(family_name)
    if fn is None:
        return transitions
    return fn(transitions, rng=rng, **kwargs)  # type: ignore[operator]
