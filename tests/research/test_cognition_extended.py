"""Extended tests for cognition research package.

These cover paths not exercised by the basic interface tests, including:
- ascending adapter channel ablation
- all world modes (native, simplified, avatar)
- dual-world env mode switching with stepping
- raw control in bodyless and embodied modes
- benchmark harness (run_baseline_suite)
- individual metric edge cases
"""

import numpy as np
import pytest

from flygym_research.cognition.adapters import AscendingAdapter
from flygym_research.cognition.body_reflex import BodylessBodyLayer
from flygym_research.cognition.config import BodyLayerConfig, EnvConfig
from flygym_research.cognition.controllers import (
    BodylessAvatarController,
    RandomController,
    RawControlController,
    ReducedDescendingController,
    ReflexOnlyController,
)
from flygym_research.cognition.envs import (
    FlyAvatarEnv,
    FlyBodyWorldEnv,
    FlyDualWorldEnv,
)
from flygym_research.cognition.experiments import (
    BenchmarkResult,
    run_baseline_suite,
    run_episode,
)
from flygym_research.cognition.interfaces import (
    AscendingSummary,
    DescendingCommand,
    RawBodyFeedback,
    RawControlCommand,
)
from flygym_research.cognition.metrics import (
    history_dependence,
    seam_fragility,
    self_world_separation,
    stabilization_quality,
    state_persistence,
    task_performance,
)
from flygym_research.cognition.worlds import (
    AvatarRemappedWorld,
    NativePhysicalWorld,
    SimplifiedEmbodiedWorld,
)

# ─── Ascending adapter ablation ────────────────────────────────────────


class TestAscendingAdapterAblation:
    def _make_raw_feedback(self) -> RawBodyFeedback:
        return RawBodyFeedback(
            time=0.01,
            joint_angles=np.zeros(6, dtype=np.float64),
            joint_velocities=np.zeros(6, dtype=np.float64),
            body_positions=np.array([[0.0, 0.0, 1.5]], dtype=np.float64),
            body_rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
            contact_active=np.ones(6, dtype=np.float64),
            contact_forces=np.zeros((6, 3), dtype=np.float64),
            contact_torques=np.zeros((6, 3), dtype=np.float64),
            contact_positions=np.zeros((6, 3), dtype=np.float64),
            contact_normals=np.zeros((6, 3), dtype=np.float64),
            contact_tangents=np.zeros((6, 3), dtype=np.float64),
            actuator_forces=np.zeros(6, dtype=np.float64),
        )

    def test_no_ablation_includes_all_channels(self):
        config = BodyLayerConfig()
        adapter = AscendingAdapter(thorax_index=0, config=config)
        adapter.reset()
        raw = self._make_raw_feedback()
        summary = adapter.summarize(raw)
        assert "stability" in summary.features
        assert "contact_fraction" in summary.features
        assert "phase" in summary.features
        assert "target_distance" in summary.features
        assert len(summary.disabled_channels) == 0

    def test_ablation_zeros_pose_channels(self):
        config = BodyLayerConfig(disabled_feedback_channels=frozenset({"pose"}))
        adapter = AscendingAdapter(thorax_index=0, config=config)
        adapter.reset()
        raw = self._make_raw_feedback()
        summary = adapter.summarize(raw)
        # Ablated channels should still exist in features but be zeroed
        assert "stability" in summary.features
        assert summary.features["stability"] == 0.0
        assert summary.features["thorax_height_mm"] == 0.0
        assert summary.features["body_speed_mm_s"] == 0.0
        # Non-ablated channels should still be present and nonzero where appropriate
        assert "contact_fraction" in summary.features
        assert "pose" in summary.disabled_channels
        # Active channels should not include the ablated keys
        assert "stability" not in summary.active_channels

    def test_ablation_zeros_contact_channels(self):
        config = BodyLayerConfig(disabled_feedback_channels=frozenset({"contact"}))
        adapter = AscendingAdapter(thorax_index=0, config=config)
        adapter.reset()
        raw = self._make_raw_feedback()
        summary = adapter.summarize(raw)
        assert summary.features["contact_fraction"] == 0.0
        assert summary.features["slip_risk"] == 0.0
        assert summary.features["collision_load"] == 0.0
        # Stability should be untouched
        assert summary.features["stability"] > 0.0

    def test_ablation_zeros_multiple_groups(self):
        config = BodyLayerConfig(
            disabled_feedback_channels=frozenset({"pose", "internal"})
        )
        adapter = AscendingAdapter(thorax_index=0, config=config)
        adapter.reset()
        raw = self._make_raw_feedback()
        summary = adapter.summarize(raw)
        assert summary.features["stability"] == 0.0
        assert summary.features["phase"] == 0.0
        assert summary.features["phase_velocity"] == 0.0
        # Feature dict should have same keys regardless of ablation
        assert "stability" in summary.features
        assert "phase" in summary.features

    def test_feature_count_constant_across_ablations(self):
        """Feature dict should always have the same set of keys."""
        raw = self._make_raw_feedback()
        no_ablation = AscendingAdapter(thorax_index=0, config=BodyLayerConfig())
        no_ablation.reset()
        ablated = AscendingAdapter(
            thorax_index=0,
            config=BodyLayerConfig(
                disabled_feedback_channels=frozenset({"pose", "contact", "internal"})
            ),
        )
        ablated.reset()
        s1 = no_ablation.summarize(raw)
        s2 = ablated.summarize(raw)
        assert set(s1.features.keys()) == set(s2.features.keys())


# ─── World modes ───────────────────────────────────────────────────────


class TestNativePhysicalWorld:
    def test_reset_and_step(self):
        world = NativePhysicalWorld()
        raw = RawBodyFeedback(
            time=0.0,
            joint_angles=np.zeros(6),
            joint_velocities=np.zeros(6),
            body_positions=np.array([[0.0, 0.0, 1.5]]),
            body_rotations=np.array([[1.0, 0.0, 0.0, 0.0]]),
            contact_active=np.ones(6),
            contact_forces=np.zeros((6, 3)),
            contact_torques=np.zeros((6, 3)),
            contact_positions=np.zeros((6, 3)),
            contact_normals=np.zeros((6, 3)),
            contact_tangents=np.zeros((6, 3)),
            actuator_forces=np.zeros(6),
        )
        summary = AscendingSummary(
            features={"stability": 0.8},
            active_channels=("stability",),
            disabled_channels=(),
        )
        ws = world.reset(seed=0, raw_feedback=raw, summary=summary)
        assert ws.mode == "native_physical"
        assert ws.step_count == 0

        raw2 = RawBodyFeedback(
            time=0.01,
            joint_angles=np.zeros(6),
            joint_velocities=np.zeros(6),
            body_positions=np.array([[0.5, 0.0, 1.5]]),
            body_rotations=np.array([[1.0, 0.0, 0.0, 0.0]]),
            contact_active=np.ones(6),
            contact_forces=np.zeros((6, 3)),
            contact_torques=np.zeros((6, 3)),
            contact_positions=np.zeros((6, 3)),
            contact_normals=np.zeros((6, 3)),
            contact_tangents=np.zeros((6, 3)),
            actuator_forces=np.zeros(6),
        )
        ws2 = world.step(DescendingCommand(), raw2, summary)
        assert ws2.step_count == 1
        assert ws2.reward > 0  # forward progress

    def test_truncation(self):
        config = EnvConfig(episode_steps=2)
        world = NativePhysicalWorld(config=config)
        raw = RawBodyFeedback(
            time=0.0,
            joint_angles=np.zeros(6),
            joint_velocities=np.zeros(6),
            body_positions=np.array([[0.0, 0.0, 1.5]]),
            body_rotations=np.array([[1.0, 0.0, 0.0, 0.0]]),
            contact_active=np.ones(6),
            contact_forces=np.zeros((6, 3)),
            contact_torques=np.zeros((6, 3)),
            contact_positions=np.zeros((6, 3)),
            contact_normals=np.zeros((6, 3)),
            contact_tangents=np.zeros((6, 3)),
            actuator_forces=np.zeros(6),
        )
        summary = AscendingSummary(
            features={"stability": 0.8},
            active_channels=("stability",),
            disabled_channels=(),
        )
        world.reset(seed=0, raw_feedback=raw, summary=summary)
        world.step(DescendingCommand(), raw, summary)
        ws = world.step(DescendingCommand(), raw, summary)
        assert ws.truncated


class TestSimplifiedEmbodiedWorld:
    def test_reset_sets_target(self):
        world = SimplifiedEmbodiedWorld()
        raw = RawBodyFeedback(
            time=0.0,
            joint_angles=np.zeros(6),
            joint_velocities=np.zeros(6),
            body_positions=np.array([[0.0, 0.0, 1.5]]),
            body_rotations=np.array([[1.0, 0.0, 0.0, 0.0]]),
            contact_active=np.ones(6),
            contact_forces=np.zeros((6, 3)),
            contact_torques=np.zeros((6, 3)),
            contact_positions=np.zeros((6, 3)),
            contact_normals=np.zeros((6, 3)),
            contact_tangents=np.zeros((6, 3)),
            actuator_forces=np.zeros(6),
        )
        summary = AscendingSummary(
            features={}, active_channels=(), disabled_channels=()
        )
        ws = world.reset(seed=42, raw_feedback=raw, summary=summary)
        assert ws.mode == "simplified_embodied"
        assert "target_vector" in ws.observables
        assert "target_xy_mm" in ws.observables

    def test_termination_on_reaching_target(self):
        config = EnvConfig(success_radius_mm=100.0)  # big radius to guarantee success
        world = SimplifiedEmbodiedWorld(config=config)
        raw = RawBodyFeedback(
            time=0.0,
            joint_angles=np.zeros(6),
            joint_velocities=np.zeros(6),
            body_positions=np.array([[0.0, 0.0, 1.5]]),
            body_rotations=np.array([[1.0, 0.0, 0.0, 0.0]]),
            contact_active=np.ones(6),
            contact_forces=np.zeros((6, 3)),
            contact_torques=np.zeros((6, 3)),
            contact_positions=np.zeros((6, 3)),
            contact_normals=np.zeros((6, 3)),
            contact_tangents=np.zeros((6, 3)),
            actuator_forces=np.zeros(6),
        )
        summary = AscendingSummary(
            features={}, active_channels=(), disabled_channels=()
        )
        world.reset(seed=1, raw_feedback=raw, summary=summary)
        # The target is placed 2-4 units away; with a 100mm radius we should
        # terminate immediately since the initial position is close enough.
        ws = world.step(DescendingCommand(), raw, summary)
        assert ws.terminated


# ─── BodylessBodyLayer raw control ─────────────────────────────────────


class TestBodylessRawControl:
    def test_step_raw_uses_command_values(self):
        body = BodylessBodyLayer()
        body.reset()
        raw_cmd = RawControlCommand(
            actuator_inputs=np.array([0.5, -0.3], dtype=np.float64)
        )
        raw_fb, summary, log = body.step_raw(raw_cmd)
        assert log["mode"] == "bodyless_raw"
        # Position should have changed based on the command
        assert not np.allclose(body._position[:2], 0.0)

    def test_step_raw_with_empty_command(self):
        body = BodylessBodyLayer()
        body.reset()
        raw_cmd = RawControlCommand(actuator_inputs=np.array([], dtype=np.float64))
        raw_fb, summary, log = body.step_raw(raw_cmd)
        assert np.allclose(body._position[:2], 0.0)


# ─── Dual world env stepping ──────────────────────────────────────────


class TestDualWorldEnvStepping:
    def test_step_in_both_modes(self):
        body = BodylessBodyLayer()
        env = FlyDualWorldEnv(
            body=body,
            worlds={
                "native": NativePhysicalWorld(),
                "avatar": AvatarRemappedWorld(),
            },
        )
        obs_native = env.reset(seed=1, mode="native")
        assert obs_native.world.mode == "native_physical"
        t1 = env.step(DescendingCommand(move_intent=0.5))
        assert t1.observation.world.mode == "native_physical"

        obs_avatar = env.reset(seed=1, mode="avatar")
        assert obs_avatar.world.mode == "avatar_remapped"
        t2 = env.step(DescendingCommand(move_intent=0.5))
        assert t2.observation.world.mode == "avatar_remapped"

    def test_invalid_mode_raises(self):
        body = BodylessBodyLayer()
        env = FlyDualWorldEnv(
            body=body,
            worlds={"native": NativePhysicalWorld()},
        )
        with pytest.raises(ValueError, match="Available modes"):
            env.set_mode("nonexistent")


# ─── Benchmark harness ─────────────────────────────────────────────────


class TestBenchmarkHarness:
    def test_run_episode_returns_transitions(self):
        env = FlyAvatarEnv(body=BodylessBodyLayer())
        controller = ReducedDescendingController()
        transitions = run_episode(env, controller, seed=1, max_steps=3)
        assert len(transitions) == 3
        assert all(hasattr(t, "reward") for t in transitions)

    def test_run_baseline_suite_produces_results(self):
        controllers = {
            "reflex": ReflexOnlyController(),
            "random": RandomController(),
        }
        results = run_baseline_suite(
            env_factory=lambda: FlyAvatarEnv(body=BodylessBodyLayer()),
            controllers=controllers,
            seeds=[0, 1],
        )
        assert len(results) == 4  # 2 controllers x 2 seeds
        assert all(isinstance(r, BenchmarkResult) for r in results)
        assert all("return" in r.metrics for r in results)

    def test_early_termination(self):
        config = EnvConfig(episode_steps=100)
        body = BodylessBodyLayer()
        # Use a world that terminates quickly
        world = SimplifiedEmbodiedWorld(config=EnvConfig(success_radius_mm=1000.0))
        env = FlyBodyWorldEnv(body=body, world=world, config=config)
        controller = ReducedDescendingController()
        transitions = run_episode(env, controller, seed=0, max_steps=100)
        # Should terminate before 100 steps due to large success radius
        assert len(transitions) < 100


# ─── Metrics edge cases ───────────────────────────────────────────────


class TestMetricEdgeCases:
    def _make_transitions(self, n: int = 10) -> list:
        env = FlyAvatarEnv(body=BodylessBodyLayer())
        controller = ReducedDescendingController()
        return run_episode(env, controller, seed=42, max_steps=n)

    def test_task_performance_empty_transitions(self):
        m = task_performance([])
        assert m["return"] == 0.0
        assert m["success"] == 0.0

    def test_stabilization_quality_nonempty(self):
        transitions = self._make_transitions(5)
        m = stabilization_quality(transitions)
        assert "stability_mean" in m
        assert "stability_min" in m
        assert m["stability_mean"] >= m["stability_min"]

    def test_state_persistence_nonempty(self):
        transitions = self._make_transitions(10)
        m = state_persistence(transitions)
        assert "state_autocorrelation" in m

    def test_state_persistence_single_step(self):
        transitions = self._make_transitions(1)
        m = state_persistence(transitions)
        assert m["state_autocorrelation"] == 0.0

    def test_history_dependence_returns_float(self):
        transitions = self._make_transitions(10)
        m = history_dependence(transitions)
        assert isinstance(m["history_dependence"], float)

    def test_seam_fragility_with_missing_log(self):
        transitions = self._make_transitions(3)
        # Bodyless mode has no mean_target_delta in body_log
        m = seam_fragility(transitions)
        assert m["seam_fragility"] == 0.0

    def test_self_world_separation_with_events(self):
        transitions = self._make_transitions(15)
        m = self_world_separation(transitions)
        assert isinstance(m["self_world_marker"], float)


# ─── Controller internal state ────────────────────────────────────────


class TestControllerState:
    def test_reflex_only_state(self):
        c = ReflexOnlyController()
        c.reset()
        assert c.get_internal_state() == {"memory": 0.0}

    def test_reduced_descending_memory_trace_updates(self):
        env = FlyAvatarEnv(body=BodylessBodyLayer())
        c = ReducedDescendingController()
        c.reset(seed=1)
        obs = env.reset(seed=1)
        c.act(obs)
        assert c.memory_trace != 0.0  # should have been updated

    def test_bodyless_avatar_inherits_from_reduced(self):
        c = BodylessAvatarController()
        c.reset(seed=0)
        env = FlyAvatarEnv(body=BodylessBodyLayer())
        obs = env.reset(seed=0)
        action = c.act(obs)
        assert isinstance(action, DescendingCommand)

    def test_raw_control_controller_produces_command(self):
        env = FlyAvatarEnv(body=BodylessBodyLayer())
        c = RawControlController()
        c.reset()
        obs = env.reset(seed=0)
        action = c.act(obs)
        assert isinstance(action, RawControlCommand)
        assert len(action.actuator_inputs) > 0


# ─── Config validation ────────────────────────────────────────────────


class TestConfig:
    def test_body_layer_config_defaults(self):
        config = BodyLayerConfig()
        assert config.timestep > 0
        assert config.gait_amplitude > 0
        assert config.bodyless_position_scale > 0

    def test_env_config_defaults(self):
        config = EnvConfig()
        assert config.episode_steps > 0
        assert config.history_length > 0
        assert config.avatar_success_radius > 0
        assert config.avatar_external_event_period > 0
