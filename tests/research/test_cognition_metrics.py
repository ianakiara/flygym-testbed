from flygym_research.cognition.controllers import ReducedDescendingController
from flygym_research.cognition.envs import FlyAvatarEnv
from flygym_research.cognition.body_reflex import BodylessBodyLayer
from flygym_research.cognition.experiments import run_episode
from flygym_research.cognition.metrics import summarize_metrics


def test_metrics_summary_contains_expected_keys():
    env = FlyAvatarEnv(body=BodylessBodyLayer())
    controller = ReducedDescendingController()
    transitions = run_episode(env, controller, seed=7, max_steps=5)
    metrics = summarize_metrics(transitions)
    assert {
        "return",
        "stability_mean",
        "state_autocorrelation",
        "history_dependence",
        "self_world_marker",
        "seam_fragility",
    } <= set(metrics)
