from copy import deepcopy

from aqa.configs import Config

LLAMA3_CONFIG = Config(
    NAME="Llama3",
    MAX_NEW_TOKENS=32
)

LLAMA3_8B_INSTRUCT_CONFIG = deepcopy(LLAMA3_CONFIG)
LLAMA3_8B_INSTRUCT_CONFIG.update(MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct")

LLAMA3_70B_INSTRUCT_CONFIG = deepcopy(LLAMA3_CONFIG)
LLAMA3_70B_INSTRUCT_CONFIG.update(MODEL_ID="meta-llama/Meta-Llama-3-70B-Instruct")

__all__ = [k for k in globals().keys() if "_CONFIG" in k]
