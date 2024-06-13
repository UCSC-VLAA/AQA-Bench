from copy import deepcopy

from .benchmark import BENCHMARK_BASE_CONFIG

BINARY_SEARCH_CONFIG = deepcopy(BENCHMARK_BASE_CONFIG)
BINARY_SEARCH_CONFIG.update(
    NAME="BinarySearch",
    MIN=32,
    MAX=32800,
)

HARD_BINARY_SEARCH_CONFIG = deepcopy(BINARY_SEARCH_CONFIG)
HARD_BINARY_SEARCH_CONFIG.update(
    MAX_STEP=30,
    MAX=32 + 2 ** 25,
)
