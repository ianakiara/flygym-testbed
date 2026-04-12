from __future__ import annotations

from ..body_reflex import BodylessBodyLayer, FlyBodyLayer
from ..config import BodyLayerConfig, EnvConfig
from ..worlds import AvatarRemappedWorld
from .fly_body_world_env import FlyBodyWorldEnv


class FlyAvatarEnv(FlyBodyWorldEnv):
    def __init__(
        self,
        *,
        body: FlyBodyLayer | BodylessBodyLayer | None = None,
        body_config: BodyLayerConfig | None = None,
        env_config: EnvConfig | None = None,
    ) -> None:
        env_config = env_config or EnvConfig()
        body = body or FlyBodyLayer(config=body_config or BodyLayerConfig())
        super().__init__(
            body=body, world=AvatarRemappedWorld(config=env_config), config=env_config
        )
