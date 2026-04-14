"""Tests for all new Phase 2/6/7/9 modules.

Covers:
- ObservationAdapter, ActionAdapter
- Tasks (navigation, target_tracking, delayed_reward, self_world_disambiguation, history_dependence)
- MemoryController, PlannerController
- New metrics (interoperability, objectness, disruption, persistence)
- Validation (claims ledger, overclaiming filter, validation suite)
"""

import numpy as np
import pytest

from flygym_research.cognition.adapters import ActionAdapter, ObservationAdapter
from flygym_research.cognition.body_reflex import BodylessBodyLayer
from flygym_research.cognition.config import EnvConfig
from flygym_research.cognition.controllers import (
    MemoryController,
    PlannerController,
    RandomController,
    ReducedDescendingController,
    SelectiveMemoryController,
)
from flygym_research.cognition.envs import FlyAvatarEnv
from flygym_research.cognition.experiments import run_episode
from flygym_research.cognition.experiments.exp_observer_interoperability import (
    _compute_raw_agreement,
    _compute_translated_agreement,
    _perturb_transitions_bias,
    _perturb_transitions_noise,
    _perturb_transitions_partial,
    _perturb_transitions_scaling,
)
from flygym_research.cognition.experiments.exp_scale_law import _summarize_stabilities
from flygym_research.cognition.interfaces import (
    AscendingSummary,
    BrainObservation,
    DescendingCommand,
    RawBodyFeedback,
    RawControlCommand,
    StepTransition,
    WorldState,
)
from flygym_research.cognition.metrics import (
    controller_action_distribution,
    cross_condition_objectness,
    cross_time_mutual_information,
    global_disruption_signature,
    hysteresis_metric,
    interoperability_score,
    latent_state_similarity,
    predictive_utility,
    reward_trajectory_similarity,
    shared_objectness_score,
    state_decay_curve,
    target_representation_stability,
)
from flygym_research.cognition.metrics.interoperability_metrics import (
    extract_state_matrix,
    translated_latent_alignment,
)
from flygym_research.cognition.tasks import (
    ConditionalSequenceTask,
    DelayedInterferenceTask,
    DelayedRewardTask,
    DistractorCueRecallTask,
    HistoryDependenceTask,
    NavigationTask,
    SelfWorldDisambiguationTask,
    TargetTrackingTask,
)
from flygym_research.cognition.validation import (
    ClaimsLedger,
    ClaimTier,
    ValidationSuite,
    overclaiming_filter,
    validate_claim_text,
)


# ─── Helpers ───────────────────────────────────────────────────────────


def _make_raw_feedback(x: float = 0.0, y: float = 0.0) -> RawBodyFeedback:
    return RawBodyFeedback(
        time=0.01,
        joint_angles=np.zeros(6, dtype=np.float64),
        joint_velocities=np.zeros(6, dtype=np.float64),
        body_positions=np.array([[x, y, 1.5]], dtype=np.float64),
        body_rotations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        contact_active=np.ones(6, dtype=np.float64),
        contact_forces=np.zeros((6, 3), dtype=np.float64),
        contact_torques=np.zeros((6, 3), dtype=np.float64),
        contact_positions=np.zeros((6, 3), dtype=np.float64),
        contact_normals=np.zeros((6, 3), dtype=np.float64),
        contact_tangents=np.zeros((6, 3), dtype=np.float64),
        actuator_forces=np.zeros(6, dtype=np.float64),
    )


def _make_summary() -> AscendingSummary:
    return AscendingSummary(
        features={
            "stability": 0.8,
            "thorax_height_mm": 1.5,
            "body_speed_mm_s": 0.3,
            "contact_fraction": 0.5,
            "slip_risk": 0.1,
            "collision_load": 0.0,
            "locomotion_quality": 0.7,
            "actuator_effort": 0.2,
            "target_distance": 2.0,
            "target_salience": 0.5,
            "phase": 0.3,
            "phase_velocity": 0.1,
        },
        active_channels=("stability", "contact", "locomotion"),
        disabled_channels=(),
    )


def _make_transitions(n: int = 10) -> list[StepTransition]:
    env = FlyAvatarEnv(body=BodylessBodyLayer())
    controller = ReducedDescendingController()
    return run_episode(env, controller, seed=42, max_steps=n)


# ─── ObservationAdapter ───────────────────────────────────────────────


class TestObservationAdapter:
    def test_compose_creates_brain_observation(self):
        adapter = ObservationAdapter()
        raw = _make_raw_feedback()
        summary = _make_summary()
        world = WorldState(
            mode="test",
            step_count=0,
            reward=0.5,
            terminated=False,
            truncated=False,
            observables={"target_vector": np.array([1.0, 0.0])},
        )
        obs = adapter.compose(raw, summary, world)
        assert isinstance(obs, BrainObservation)
        assert obs.world.mode == "test"
        assert len(obs.history) == 1

    def test_history_accumulates(self):
        adapter = ObservationAdapter(config=EnvConfig(history_length=4))
        raw = _make_raw_feedback()
        summary = _make_summary()
        world = WorldState(
            mode="test", step_count=0, reward=0.0,
            terminated=False, truncated=False, observables={},
        )
        for i in range(6):
            obs = adapter.compose(raw, summary, world)
        assert len(obs.history) == 4  # Capped at history_length.

    def test_flatten_observation(self):
        adapter = ObservationAdapter()
        raw = _make_raw_feedback()
        summary = _make_summary()
        world = WorldState(
            mode="test", step_count=0, reward=1.0,
            terminated=False, truncated=False,
            observables={"val": 0.5},
        )
        obs = adapter.compose(raw, summary, world)
        flat = adapter.flatten_observation(obs)
        assert isinstance(flat, np.ndarray)
        assert flat.ndim == 1
        assert len(flat) > 0

    def test_reset_clears_history(self):
        adapter = ObservationAdapter()
        raw = _make_raw_feedback()
        summary = _make_summary()
        world = WorldState(
            mode="test", step_count=0, reward=0.0,
            terminated=False, truncated=False, observables={},
        )
        adapter.compose(raw, summary, world)
        adapter.reset()
        obs = adapter.compose(raw, summary, world)
        assert len(obs.history) == 1

    def test_observation_spec(self):
        adapter = ObservationAdapter()
        spec = adapter.observation_spec()
        assert "history_length" in spec
        assert "raw_body_fields" in spec


# ─── ActionAdapter ─────────────────────────────────────────────────────


class TestActionAdapter:
    def test_from_descending_clips(self):
        adapter = ActionAdapter(clip=True)
        cmd = DescendingCommand(move_intent=5.0, turn_intent=-5.0)
        clipped = adapter.from_descending(cmd)
        assert clipped.move_intent == 1.0
        assert clipped.turn_intent == -1.0

    def test_from_descending_no_clip(self):
        adapter = ActionAdapter(clip=False)
        cmd = DescendingCommand(move_intent=5.0)
        result = adapter.from_descending(cmd)
        assert result.move_intent == 5.0

    def test_from_dict(self):
        adapter = ActionAdapter()
        cmd = adapter.from_dict({"move_intent": 0.5, "turn_intent": -0.3})
        assert isinstance(cmd, DescendingCommand)
        assert abs(cmd.move_intent - 0.5) < 1e-6
        assert abs(cmd.turn_intent - (-0.3)) < 1e-6

    def test_from_flat(self):
        adapter = ActionAdapter()
        vec = np.array([0.5, -0.3, 0.0, 0.8, 0.1, 0.0])
        cmd = adapter.from_flat(vec)
        assert abs(cmd.move_intent - 0.5) < 1e-6
        assert abs(cmd.stabilization_priority - 0.8) < 1e-6

    def test_to_flat(self):
        adapter = ActionAdapter()
        cmd = DescendingCommand(move_intent=0.5, turn_intent=-0.3)
        flat = adapter.to_flat(cmd)
        assert len(flat) == 6
        assert abs(flat[0] - 0.5) < 1e-6

    def test_from_raw_passthrough(self):
        adapter = ActionAdapter()
        raw = RawControlCommand(actuator_inputs=np.array([1.0, 2.0]))
        result = adapter.from_raw(raw)
        assert np.allclose(result.actuator_inputs, raw.actuator_inputs)

    def test_action_spec(self):
        adapter = ActionAdapter()
        spec = adapter.action_spec()
        assert "move_intent" in spec
        assert spec["move_intent"] == (-1.0, 1.0)


# ─── Tasks ─────────────────────────────────────────────────────────────


class TestNavigationTask:
    def test_reset_and_step(self):
        task = NavigationTask()
        raw = _make_raw_feedback()
        summary = _make_summary()
        ws = task.reset(seed=0, raw_feedback=raw, summary=summary)
        assert ws.mode == "navigation_task"
        assert "target_vector" in ws.observables

        ws2 = task.step(DescendingCommand(), raw, summary)
        assert ws2.step_count == 1

    def test_termination_on_reaching_target(self):
        config = EnvConfig(success_radius_mm=1000.0)
        task = NavigationTask(config=config)
        raw = _make_raw_feedback()
        summary = _make_summary()
        task.reset(seed=0, raw_feedback=raw, summary=summary)
        ws = task.step(DescendingCommand(), raw, summary)
        assert ws.terminated  # Large radius should terminate immediately.


class TestTargetTrackingTask:
    def test_reset_and_step(self):
        task = TargetTrackingTask()
        raw = _make_raw_feedback()
        summary = _make_summary()
        ws = task.reset(seed=0, raw_feedback=raw, summary=summary)
        assert ws.mode == "target_tracking_task"

        ws2 = task.step(DescendingCommand(), raw, summary)
        assert ws2.step_count == 1
        assert not ws2.terminated  # Tracking tasks don't terminate early.


class TestDelayedRewardTask:
    def test_reset_and_step(self):
        task = DelayedRewardTask()
        raw = _make_raw_feedback()
        summary = _make_summary()
        ws = task.reset(seed=0, raw_feedback=raw, summary=summary)
        assert ws.mode == "delayed_reward_task"

    def test_delayed_reward_delivery(self):
        config = EnvConfig(success_radius_mm=1000.0, episode_steps=20)
        task = DelayedRewardTask(config=config, reward_delay=3)
        raw = _make_raw_feedback()
        summary = _make_summary()
        task.reset(seed=0, raw_feedback=raw, summary=summary)

        rewards = []
        for _ in range(10):
            ws = task.step(DescendingCommand(), raw, summary)
            rewards.append(ws.reward)

        # Should have a reward pulse at step 1 (reached) + 3 (delay) = step 4.
        assert any(r > 0 for r in rewards)


class TestSelfWorldDisambiguationTask:
    def test_reset_and_step(self):
        task = SelfWorldDisambiguationTask()
        raw = _make_raw_feedback()
        summary = _make_summary()
        ws = task.reset(seed=0, raw_feedback=raw, summary=summary)
        assert ws.mode == "self_world_disambiguation_task"

    def test_perturbation_occurs(self):
        task = SelfWorldDisambiguationTask(perturbation_period=2)
        raw = _make_raw_feedback()
        summary = _make_summary()
        task.reset(seed=0, raw_feedback=raw, summary=summary)

        perturbations = []
        for _ in range(6):
            ws = task.step(DescendingCommand(), raw, summary)
            perturbations.append(ws.info.get("perturbation_active", False))
        assert any(perturbations)


class TestHistoryDependenceTask:
    def test_reset_and_step(self):
        task = HistoryDependenceTask()
        raw = _make_raw_feedback()
        summary = _make_summary()
        ws = task.reset(seed=0, raw_feedback=raw, summary=summary)
        assert ws.mode == "history_dependence_task"
        assert "waypoints" in ws.observables
        assert "visited" in ws.observables


class TestSelectiveMemoryTasks:
    def test_distractor_cue_recall_reset_and_step(self):
        task = DistractorCueRecallTask()
        raw = _make_raw_feedback()
        summary = _make_summary()
        ws = task.reset(seed=0, raw_feedback=raw, summary=summary)
        assert ws.mode == "distractor_cue_recall_task"
        assert "cue_vector" in ws.observables
        ws2 = task.step(DescendingCommand(), raw, summary)
        assert "distractor_active" in ws2.info

    def test_conditional_sequence_reset_and_step(self):
        task = ConditionalSequenceTask()
        raw = _make_raw_feedback()
        summary = _make_summary()
        ws = task.reset(seed=0, raw_feedback=raw, summary=summary)
        assert ws.mode == "conditional_sequence_task"
        assert "context_key" in ws.observables
        ws2 = task.step(DescendingCommand(), raw, summary)
        assert "required_order" in ws2.info

    def test_delayed_interference_reset_and_step(self):
        task = DelayedInterferenceTask()
        raw = _make_raw_feedback()
        summary = _make_summary()
        ws = task.reset(seed=0, raw_feedback=raw, summary=summary)
        assert ws.mode == "delayed_interference_task"
        ws2 = task.step(DescendingCommand(), raw, summary)
        assert "reward_delay" in ws2.info


# ─── MemoryController ─────────────────────────────────────────────────


class TestMemoryController:
    def test_reset_and_act(self):
        env = FlyAvatarEnv(body=BodylessBodyLayer())
        ctrl = MemoryController()
        ctrl.reset(seed=0)
        obs = env.reset(seed=0)
        action = ctrl.act(obs)
        assert isinstance(action, DescendingCommand)

    def test_internal_state_updates(self):
        env = FlyAvatarEnv(body=BodylessBodyLayer())
        ctrl = MemoryController()
        ctrl.reset(seed=0)
        obs = env.reset(seed=0)
        ctrl.act(obs)
        state = ctrl.get_internal_state()
        assert state["step_count"] == 1.0
        assert state["memory_length"] == 1.0

    def test_memory_accumulates(self):
        env = FlyAvatarEnv(body=BodylessBodyLayer())
        ctrl = MemoryController(memory_size=4)
        ctrl.reset(seed=0)
        obs = env.reset(seed=0)
        for _ in range(6):
            action = ctrl.act(obs)
            t = env.step(action)
            obs = t.observation
        state = ctrl.get_internal_state()
        assert state["memory_length"] == 4.0

    def test_hidden_state_nonzero(self):
        env = FlyAvatarEnv(body=BodylessBodyLayer())
        ctrl = MemoryController()
        ctrl.reset(seed=0)
        obs = env.reset(seed=0)
        ctrl.act(obs)
        assert ctrl._hidden is not None
        assert not np.allclose(ctrl._hidden, 0.0)


class TestSelectiveMemoryController:
    def test_reset_and_act(self):
        env = FlyAvatarEnv(body=BodylessBodyLayer())
        ctrl = SelectiveMemoryController()
        ctrl.reset(seed=0)
        obs = env.reset(seed=0)
        action = ctrl.act(obs)
        assert isinstance(action, DescendingCommand)

    def test_internal_state_updates(self):
        env = FlyAvatarEnv(body=BodylessBodyLayer())
        ctrl = SelectiveMemoryController()
        ctrl.reset(seed=0)
        obs = env.reset(seed=0)
        ctrl.act(obs)
        state = ctrl.get_internal_state()
        assert state["step_count"] == 1.0
        assert state["active_slots"] >= 1.0

    @pytest.mark.parametrize(
        ("context_key", "expected_slot"),
        [(0.5, 0), (1.0, 0), (1.5, 1), (2.5, 2), (3.5, 2)],
    )
    def test_cue_slot_boundaries_map_consistently(self, context_key, expected_slot):
        ctrl = SelectiveMemoryController()
        assert ctrl._cue_slot_index(context_key) == expected_slot

    def test_neutral_writes_do_not_corrupt_cue_slots(self):
        env = FlyAvatarEnv(body=BodylessBodyLayer())
        ctrl = SelectiveMemoryController()
        ctrl.reset(seed=0)
        obs = env.reset(seed=0)

        cue_obs = BrainObservation(
            raw_body=obs.raw_body,
            summary=obs.summary,
            world=WorldState(
                mode=obs.world.mode,
                step_count=obs.world.step_count,
                reward=obs.world.reward,
                terminated=obs.world.terminated,
                truncated=obs.world.truncated,
                observables={
                    **obs.world.observables,
                    "context_key": 1.0,
                    "cue_vector": np.array([0.8, -0.6], dtype=np.float64),
                },
                info=dict(obs.world.info),
            ),
            history=obs.history,
        )
        cue_query = ctrl._build_query(cue_obs)
        ctrl._write(cue_query, cue_obs, np.zeros(ctrl.memory_slots, dtype=np.float64))
        cue_slot_before = ctrl._slots[0].copy()

        ctrl._strengths[:] = 1.0
        spare_slot_before = ctrl._slots[3].copy()

        neutral_obs = BrainObservation(
            raw_body=obs.raw_body,
            summary=obs.summary,
            world=WorldState(
                mode=obs.world.mode,
                step_count=obs.world.step_count,
                reward=obs.world.reward,
                terminated=obs.world.terminated,
                truncated=obs.world.truncated,
                observables={
                    **obs.world.observables,
                    "context_key": 0.0,
                    "cue_vector": np.array([0.1, 0.2], dtype=np.float64),
                },
                info={**obs.world.info, "distractor_active": False},
            ),
            history=obs.history,
        )
        neutral_query = ctrl._build_query(neutral_obs)
        ctrl._write(neutral_query, neutral_obs, np.zeros(ctrl.memory_slots, dtype=np.float64))

        assert np.allclose(ctrl._slots[0], cue_slot_before)
        assert not np.allclose(ctrl._slots[3], spare_slot_before)


# ─── PlannerController ────────────────────────────────────────────────


class TestPlannerController:
    def test_reset_and_act(self):
        env = FlyAvatarEnv(body=BodylessBodyLayer())
        ctrl = PlannerController()
        ctrl.reset(seed=0)
        obs = env.reset(seed=0)
        action = ctrl.act(obs)
        assert isinstance(action, DescendingCommand)

    def test_plan_generated(self):
        env = FlyAvatarEnv(body=BodylessBodyLayer())
        ctrl = PlannerController(num_subgoals=3)
        ctrl.reset(seed=0)
        obs = env.reset(seed=0)
        ctrl.act(obs)
        state = ctrl.get_internal_state()
        assert state["total_replans"] >= 1.0

    def test_subgoal_advancement(self):
        env = FlyAvatarEnv(body=BodylessBodyLayer())
        ctrl = PlannerController(num_subgoals=2, subgoal_timeout=2)
        ctrl.reset(seed=0)
        obs = env.reset(seed=0)
        for _ in range(5):
            action = ctrl.act(obs)
            t = env.step(action)
            obs = t.observation
        # After 5 steps with timeout=2, subgoals should have advanced.
        state = ctrl.get_internal_state()
        assert state["step_count"] == 5.0


# ─── Interoperability metrics ─────────────────────────────────────────


class TestInteroperabilityMetrics:
    def test_action_distribution(self):
        transitions = _make_transitions(10)
        result = controller_action_distribution(transitions)
        assert "action_move_mean" in result
        assert "action_turn_std" in result

    def test_latent_state_similarity(self):
        t1 = _make_transitions(10)
        t2 = _make_transitions(10)
        result = latent_state_similarity(t1, t2)
        assert "latent_correlation" in result
        assert "latent_mae" in result

    def test_reward_trajectory_similarity(self):
        t1 = _make_transitions(10)
        t2 = _make_transitions(10)
        result = reward_trajectory_similarity(t1, t2)
        assert "reward_correlation" in result

    def test_interoperability_score_composite(self):
        t1 = _make_transitions(10)
        t2 = _make_transitions(10)
        result = interoperability_score(t1, t2)
        assert "interoperability_score" in result
        assert 0.0 <= result["interoperability_score"] <= 1.0

    def test_empty_transitions(self):
        result = controller_action_distribution([])
        assert result["action_move_mean"] == 0.0

    def test_translated_alignment_residual_tracks_selected_direction(self, monkeypatch):
        import flygym_research.cognition.metrics.interoperability_metrics as interop_metrics

        sentinel_a = object()
        sentinel_b = object()
        transitions_a = [sentinel_a] * 8
        transitions_b = [sentinel_b] * 8
        matrix_a = np.array(
            [[0.0, 0.0], [1.0, 1.0], [2.0, 4.0], [3.0, 9.0], [4.0, 16.0], [5.0, 25.0], [6.0, 36.0], [7.0, 49.0]],
            dtype=np.float64,
        )
        matrix_b = np.array(
            [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0], [5.0, 1.0], [6.0, 1.0], [7.0, 1.0]],
            dtype=np.float64,
        )

        def fake_extract_state_matrix(transitions):
            return matrix_a if transitions and transitions[0] is sentinel_a else matrix_b

        monkeypatch.setattr(interop_metrics, "extract_state_matrix", fake_extract_state_matrix)
        result = translated_latent_alignment(transitions_a, transitions_b)
        assert result["translation_r2_ab"] == pytest.approx(1.0)
        assert result["translation_r2_ba"] < result["translation_r2_ab"]
        assert result["translation_residual_norm"] == pytest.approx(0.0)


class TestObserverInteropExperiment:
    def test_noise_perturbation_changes_measured_state(self):
        transitions = _make_transitions(10)
        perturbed = _perturb_transitions_noise(
            transitions,
            scale=0.4,
            rng=np.random.default_rng(0),
        )
        assert not np.allclose(
            extract_state_matrix(transitions),
            extract_state_matrix(perturbed),
        )
        raw = _compute_raw_agreement(transitions, perturbed)
        assert raw["raw_mse"] > 0.0

    def test_scaling_perturbation_changes_measured_state(self):
        transitions = _make_transitions(10)
        perturbed = _perturb_transitions_scaling(transitions, factor=1.8)
        raw = _compute_raw_agreement(transitions, perturbed)
        assert raw["raw_mse"] > 0.0

    def test_partial_observation_changes_measured_state(self):
        transitions = _make_transitions(10)
        perturbed = _perturb_transitions_partial(
            transitions,
            drop_fraction=0.5,
            rng=np.random.default_rng(0),
        )
        raw = _compute_raw_agreement(transitions, perturbed)
        assert raw["raw_mse"] > 0.0

    def test_translation_improves_over_raw_for_bias_perturbation(self):
        transitions = _make_transitions(10)
        perturbed = _perturb_transitions_bias(transitions, bias=1.5)
        raw = _compute_raw_agreement(transitions, perturbed)
        translated = _compute_translated_agreement(transitions, perturbed)
        assert raw["raw_mse"] > translated["translated_mse"]


class TestScaleLawHelpers:
    def test_stability_summary_keeps_infinite_coefficients_unstable(self):
        summary = _summarize_stabilities([float("inf"), 0.1, 0.2])
        assert summary["mean_stability_cv"] == pytest.approx(0.15)
        assert not summary["is_stable"]


# ─── Objectness metrics ───────────────────────────────────────────────


class TestObjectnessMetrics:
    def test_target_representation_stability(self):
        transitions = _make_transitions(10)
        result = target_representation_stability(transitions)
        assert "target_distance_cv" in result
        assert "target_persistence" in result

    def test_cross_condition_objectness(self):
        t1 = _make_transitions(10)
        t2 = _make_transitions(10)
        result = cross_condition_objectness(t1, t2)
        assert "objectness_persistence_diff" in result

    def test_shared_objectness_score(self):
        t1 = _make_transitions(10)
        t2 = _make_transitions(10)
        result = shared_objectness_score(t1, t2)
        assert "shared_objectness_score" in result
        assert 0.0 <= result["shared_objectness_score"] <= 1.0


# ─── Disruption metrics ───────────────────────────────────────────────


class TestDisruptionMetrics:
    def test_global_disruption_signature(self):
        baseline = _make_transitions(10)
        # Use different seed for "disrupted" condition.
        env = FlyAvatarEnv(body=BodylessBodyLayer())
        ctrl = RandomController()
        disrupted = run_episode(env, ctrl, seed=99, max_steps=10)
        result = global_disruption_signature(baseline, disrupted)
        assert "fragmentation_score" in result
        assert "n_metrics_degraded" in result
        assert "is_global_disruption" in result


# ─── Persistence metrics ──────────────────────────────────────────────


class TestPersistenceMetrics:
    def test_cross_time_mi(self):
        transitions = _make_transitions(15)
        result = cross_time_mutual_information(transitions, max_lag=3)
        assert "mi_mean" in result
        assert "mi_lag_1" in result

    def test_state_decay_curve(self):
        transitions = _make_transitions(15)
        result = state_decay_curve(transitions, max_lag=5)
        assert "decay_half_life" in result
        assert "autocorr_lag_1" in result

    def test_predictive_utility(self):
        transitions = _make_transitions(10)
        result = predictive_utility(transitions, horizon=2)
        assert "predictive_utility" in result
        assert result["predictive_utility"] >= 0.0

    def test_hysteresis_metric(self):
        transitions = _make_transitions(10)
        result = hysteresis_metric(transitions)
        assert "hysteresis_score" in result
        assert result["hysteresis_score"] >= 0.0

    def test_short_transitions(self):
        transitions = _make_transitions(2)
        mi = cross_time_mutual_information(transitions, max_lag=3)
        assert mi["mi_mean"] == 0.0
        decay = state_decay_curve(transitions, max_lag=5)
        assert decay["decay_half_life"] == 0.0


# ─── Claims ledger ────────────────────────────────────────────────────


class TestClaimsLedger:
    def test_register_useful_claim(self):
        ledger = ClaimsLedger()
        claim = ledger.register(
            text="Body substrate is a candidate prerequisite for state persistence",
            tier=ClaimTier.USEFUL,
            experiment="Exp 1",
            evidence=["Embodied agents show 2x higher autocorrelation"],
        )
        assert claim.claim_id == "CLM-0001"
        assert claim.tier == ClaimTier.USEFUL
        assert len(ledger.claims) == 1

    def test_register_strong_claim_requires_ablations(self):
        ledger = ClaimsLedger()
        with pytest.raises(ValueError, match="survived ablations"):
            ledger.register(
                text="Stability feedback is a control integration marker",
                tier=ClaimTier.STRONG,
                experiment="Exp 3",
                evidence=["40% degradation on ablation"],
            )

    def test_register_strong_claim_requires_failure_modes(self):
        ledger = ClaimsLedger()
        with pytest.raises(ValueError, match="failure modes"):
            ledger.register(
                text="Stability feedback is a control integration marker",
                tier=ClaimTier.STRONG,
                experiment="Exp 3",
                evidence=["40% degradation on ablation"],
                ablation_survived=["contact ablation"],
            )

    def test_register_promoted_claim_requires_negative_controls(self):
        ledger = ClaimsLedger()
        with pytest.raises(ValueError, match="negative control"):
            ledger.register(
                text="Stability feedback is a control integration marker",
                tier=ClaimTier.PROMOTED,
                experiment="Exp 3",
                evidence=["40% degradation"],
                ablation_survived=["contact", "effort"],
                failure_modes=["Memory compensates"],
            )

    def test_overclaiming_filter_blocks_forbidden_language(self):
        ledger = ClaimsLedger()
        with pytest.raises(ValueError, match="forbidden language"):
            ledger.register(
                text="The fly is conscious",
                tier=ClaimTier.USEFUL,
                experiment="Exp 1",
                evidence=["test"],
            )

    def test_to_markdown(self):
        ledger = ClaimsLedger()
        ledger.register(
            text="Body substrate is a candidate prerequisite",
            tier=ClaimTier.USEFUL,
            experiment="Exp 1",
            evidence=["Evidence 1"],
        )
        md = ledger.to_markdown()
        assert "Claims Ledger" in md
        assert "candidate prerequisite" in md

    def test_to_json(self):
        ledger = ClaimsLedger()
        ledger.register(
            text="Test claim",
            tier=ClaimTier.USEFUL,
            experiment="Test",
            evidence=["Evidence"],
        )
        j = ledger.to_json()
        assert "Test claim" in j

    def test_filter_by_tier(self):
        ledger = ClaimsLedger()
        ledger.register(
            text="Useful claim",
            tier=ClaimTier.USEFUL,
            experiment="Exp1",
            evidence=["e1"],
        )
        ledger.register(
            text="Strong claim with a self/world separation marker",
            tier=ClaimTier.STRONG,
            experiment="Exp3",
            evidence=["e2"],
            ablation_survived=["a1"],
            failure_modes=["f1"],
        )
        useful = ledger.filter_by_tier(ClaimTier.USEFUL)
        assert len(useful) == 1


class TestOverclaimingFilter:
    def test_clean_text(self):
        violations = overclaiming_filter("Body substrate is a candidate prerequisite")
        assert violations == []

    def test_detects_forbidden(self):
        violations = overclaiming_filter("We found consciousness in the fly")
        assert len(violations) > 0

    def test_validate_claim_text(self):
        result = validate_claim_text(
            "This is a candidate prerequisite for minimal cognition"
        )
        assert result["valid"] is True
        assert result["uses_approved_vocabulary"] is True


# ─── ValidationSuite ──────────────────────────────────────────────────


class TestValidationSuite:
    def test_beats_baseline(self):
        suite = ValidationSuite()
        # Both from same controller — should be roughly equal.
        t1 = _make_transitions(10)
        t2 = _make_transitions(10)
        result = suite.check_beats_baseline(t1, t2, metric_key="return")
        assert isinstance(result.passed, bool)
        assert "beats_baseline" in result.check_name

    def test_ablation_survival(self):
        suite = ValidationSuite()
        intact = _make_transitions(10)
        ablated = _make_transitions(10)
        result = suite.check_ablation_survival(intact, ablated)
        assert isinstance(result.passed, bool)

    def test_negative_control(self):
        suite = ValidationSuite()
        experimental = _make_transitions(10)
        env = FlyAvatarEnv(body=BodylessBodyLayer())
        neg = run_episode(env, RandomController(), seed=0, max_steps=10)
        result = suite.check_negative_control(neg, experimental)
        assert isinstance(result.passed, bool)

    def test_summary(self):
        suite = ValidationSuite()
        t1 = _make_transitions(10)
        t2 = _make_transitions(10)
        suite.check_beats_baseline(t1, t2)
        s = suite.summary()
        assert s["total_checks"] == 1

    def test_to_markdown(self):
        suite = ValidationSuite()
        t1 = _make_transitions(10)
        t2 = _make_transitions(10)
        suite.check_beats_baseline(t1, t2)
        md = suite.to_markdown()
        assert "Validation Results" in md


# ─── Causal intervention metrics ─────────────────────────────────────


class TestCausalMetrics:
    def test_causal_influence_score(self):
        from flygym_research.cognition.metrics.causal_metrics import causal_influence_score

        baseline = _make_transitions(15)
        # Use different seed to produce divergent trajectory.
        env2 = FlyAvatarEnv(body=BodylessBodyLayer())
        ctrl2 = MemoryController()
        intervened = run_episode(env2, ctrl2, seed=99, max_steps=15)

        result = causal_influence_score(baseline, intervened, intervention_magnitude=0.5)
        assert "causal_influence" in result
        assert 0.0 <= result["causal_influence"] <= 1.0
        assert result["action_divergence"] >= 0.0
        assert result["trajectory_divergence"] >= 0.0

    def test_causal_influence_empty(self):
        from flygym_research.cognition.metrics.causal_metrics import causal_influence_score

        result = causal_influence_score([], [], intervention_magnitude=1.0)
        assert result["causal_influence"] == 0.0

    def test_epiphenomenal_test(self):
        from flygym_research.cognition.metrics.causal_metrics import epiphenomenal_test

        baseline = _make_transitions(15)
        result = epiphenomenal_test(baseline, baseline)
        assert "epiphenomenal_score" in result
        # Identical trajectories → state is "epiphenomenal" (unchanged).
        assert result["epiphenomenal_score"] == 1.0

    def test_temporal_causal_depth(self):
        from flygym_research.cognition.metrics.causal_metrics import temporal_causal_depth

        transitions = _make_transitions(25)
        result = temporal_causal_depth(transitions, max_horizon=5)
        assert "causal_depth" in result
        assert "causal_depth_score" in result
        assert 0.0 <= result["causal_depth_score"] <= 1.0

    def test_temporal_causal_depth_short(self):
        from flygym_research.cognition.metrics.causal_metrics import temporal_causal_depth

        result = temporal_causal_depth([], max_horizon=5)
        assert result["causal_depth"] == 0.0


# ─── Causal intervention experiment ──────────────────────────────────


class TestCausalInterventionExperiment:
    def test_run_experiment(self, tmp_path):
        from flygym_research.cognition.experiments.exp_causal_intervention import run_experiment

        result = run_experiment(tmp_path / "causal")
        assert "summary" in result
        summary = result["summary"]
        assert "mean_causal_influence" in summary
        assert "mean_epiphenomenal_score" in summary
        assert "mean_causal_depth" in summary
        assert (tmp_path / "causal" / "causal_intervention_results.json").exists()


# ─── Claims ledger round-trip ─────────────────────────────────────────


class TestClaimsLedgerRoundTrip:
    def test_from_json_preserves_promoted_from(self):
        ledger = ClaimsLedger()
        claim = ledger.register(
            text="Candidate prerequisite: persistent state exists",
            tier=ClaimTier.USEFUL,
            experiment="test",
            evidence=["data"],
        )
        # Promote the claim.
        ledger.promote(claim.claim_id, ClaimTier.STRONG)
        json_str = ledger.to_json()

        restored = ClaimsLedger.from_json(json_str)
        restored_claim = restored.get(claim.claim_id)
        assert restored_claim is not None
        assert restored_claim.tier == ClaimTier.STRONG
        assert restored_claim.promoted_from == ClaimTier.USEFUL

    def test_save_and_load(self, tmp_path):
        ledger = ClaimsLedger()
        ledger.register(
            text="Candidate prerequisite: test claim",
            tier=ClaimTier.USEFUL,
            experiment="exp",
            evidence=["ev1"],
        )
        ledger.save(tmp_path / "ledger")
        loaded = ClaimsLedger.load(tmp_path / "ledger")
        assert len(loaded.claims) == 1
        assert loaded.claims[0].text == "Candidate prerequisite: test claim"
