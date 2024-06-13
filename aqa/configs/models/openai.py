from copy import deepcopy

from aqa.configs import Config


OPENAI_CONFIG = Config(
    NAME="OpenAI",
    API_VERSION="2023-12-01-preview",
    SLEEP_SEC=0.5
)

__all__ = [k for k in globals().keys() if "_CONFIG" in k]
