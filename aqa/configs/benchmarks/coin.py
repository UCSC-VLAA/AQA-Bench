from copy import deepcopy

from .benchmark import BENCHMARK_BASE_CONFIG

COIN_CONFIG = deepcopy(BENCHMARK_BASE_CONFIG)
COIN_CONFIG.update(
    NAME="Coin",
    MIN=32,
    MAX=32800,
)

HARD_COIN_CONFIG = deepcopy(COIN_CONFIG)
HARD_COIN_CONFIG.update(
    MAX_STEP=30,
    MAX=32 + 2 ** 25,
)
