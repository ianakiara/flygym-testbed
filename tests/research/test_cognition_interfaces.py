import numpy as np

from flygym_research.cognition.controllers import (
    NoAscendingFeedbackController,
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
from flygym_research.cognition.body_reflex import (
    BodyLayerConfig,
    BodylessBodyLayer,
    FlyBodyLayer,
)
from flygym_research.cognition.interfaces import DescendingCommand, RawControlCommand
from flygym_research.cognition.worlds import (
    AvatarRemappedWorld,
    NativePhysicalWorld,
    SimplifiedEmbodiedWorld,
)


def test_bodyless_avatar_env_reset_and_step():
    env = FlyAvatarEnv(body=BodylessBodyLayer())
    observation = env.reset(seed=3)
    assert observation.world.mode == "avatar_remapped"
    transition = env.step(DescendingCommand(move_intent=0.5, turn_intent=-0.2))
    assert "target_vector" in transition.observation.world.observables
    assert transition.reward <= 1.0


def test_dual_world_switch_preserves_body_layer():
    body = BodylessBodyLayer()
    env = FlyDualWorldEnv(
        body=body,
        worlds={
            "native": NativePhysicalWorld(),
            "avatar": AvatarRemappedWorld(),
        },
    )
    env.reset(seed=1, mode="native")
    env.set_mode("avatar")
    observation = env.reset(seed=1)
    assert observation.world.mode == "avatar_remapped"
    assert env.body is body


def test_reduced_and_raw_controllers_emit_expected_command_types():
    env = FlyAvatarEnv(body=BodylessBodyLayer())
    observation = env.reset(seed=4)
    reduced = ReducedDescendingController()
    raw = RawControlController()
    reduced.reset(seed=4)
    raw.reset(seed=4)
    assert isinstance(reduced.act(observation), DescendingCommand)
    assert isinstance(raw.act(observation), RawControlCommand)


def test_baseline_controllers_are_callable():
    env = FlyAvatarEnv(body=BodylessBodyLayer())
    observation = env.reset(seed=2)
    controllers = [
        ReflexOnlyController(),
        RandomController(),
        ReducedDescendingController(),
        NoAscendingFeedbackController(),
    ]
    for controller in controllers:
        controller.reset(seed=2)
        action = controller.act(observation)
        assert hasattr(action, "metadata")


def test_embodied_env_runs_one_step_with_real_body():
    body = FlyBodyLayer(config=BodyLayerConfig(warmup_duration_s=0.001))
    env = FlyBodyWorldEnv(body=body, world=SimplifiedEmbodiedWorld())
    observation = env.reset(seed=5)
    transition = env.step(DescendingCommand(move_intent=0.2, turn_intent=0.1))
    assert observation.summary.features["stability"] >= 0.0
    assert np.isfinite(transition.reward)
