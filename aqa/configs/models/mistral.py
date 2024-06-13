from copy import deepcopy

from aqa.configs import Config

MISTRAL_CONFIG = Config(
    NAME="Mistral",
    MAX_NEW_TOKENS=128
)

MISTRAL_7B_INSTRUCT_v01_CONFIG = deepcopy(MISTRAL_CONFIG)
MISTRAL_7B_INSTRUCT_v01_CONFIG.update(MODEL_ID="mistralai/Mistral-7B-Instruct-v0.1")

MISTRAL_7B_INSTRUCT_v02_CONFIG = deepcopy(MISTRAL_CONFIG)
MISTRAL_7B_INSTRUCT_v02_CONFIG.update(MODEL_ID="mistralai/Mistral-7B-Instruct-v0.2")

MIXTRAL_8x7B_INSTRUCT_v01_CONFIG = deepcopy(MISTRAL_CONFIG)
MIXTRAL_8x7B_INSTRUCT_v01_CONFIG.update(MODEL_ID="mistralai/Mixtral-8x7B-Instruct-v0.1")

__all__ = [k for k in globals().keys() if "_CONFIG" in k]
