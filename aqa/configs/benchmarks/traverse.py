from copy import deepcopy

from .benchmark import BENCHMARK_BASE_CONFIG

BFS_CONFIG = deepcopy(BENCHMARK_BASE_CONFIG)
BFS_CONFIG.update(
    NAME="BFS",
    NODE_NUM=15,
    EXPLAIN_ALGO=True,
    MCQ=False,
    PROVIDE_STATE=False,
)

DFS_CONFIG = deepcopy(BENCHMARK_BASE_CONFIG)
DFS_CONFIG.update(
    NAME="DFS",
    NODE_NUM=8,
    EXPLAIN_ALGO=True,
    MCQ=False,
    PROVIDE_STATE=False,
)

HARD_BFS_CONFIG = deepcopy(BFS_CONFIG)
HARD_BFS_CONFIG.update(
    MAX_STEP=30,
    NODE_NUM=25,
)

HARD_DFS_CONFIG = deepcopy(DFS_CONFIG)
HARD_DFS_CONFIG.update(
    MAX_STEP=30,
    NODE_NUM=13,
)
