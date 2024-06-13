from aqa.configs import Config


GEMINI_CONFIG = Config(
    NAME="Gemini",
    SLEEP_SEC=0.2
)

__all__ = [k for k in globals().keys() if "_CONFIG" in k]
