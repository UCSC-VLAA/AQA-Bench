from copy import deepcopy

from aqa.configs import Config

DEEPSEEK_CONFIG = Config(
    NAME="Deepseek",
    MAX_NEW_TOKENS=128
)

DEEPSEEK_LLM_7B_CONFIG = deepcopy(DEEPSEEK_CONFIG)
DEEPSEEK_LLM_7B_CONFIG.update(MODEL_ID="deepseek-ai/deepseek-llm-7b-chat")

DEEPSEEK_LLM_67B_CONFIG = deepcopy(DEEPSEEK_CONFIG)
DEEPSEEK_LLM_67B_CONFIG.update(MODEL_ID="deepseek-ai/deepseek-llm-67b-chat")

DEEPSEEK_MOE_16B_CONFIG = deepcopy(DEEPSEEK_CONFIG)
DEEPSEEK_MOE_16B_CONFIG.update(MODEL_ID="deepseek-ai/deepseek-moe-16b-chat")

__all__ = [k for k in globals().keys() if "_CONFIG" in k]
