from copy import deepcopy
import os.path as osp

from aqa.configs import benchmarks, models
from aqa.configs.base_config import BASE_CONFIG

config = deepcopy(BASE_CONFIG)
config.update(
    BENCHMARK=dict(
        benchmarks.DFS_CONFIG,
        DATASET_FILE="dfs_0.json",
        SAVE_PERIOD=25,
        EXP_NAME=osp.splitext(osp.split(__file__)[1])[0],
        OUTPUT_DIR=osp.split(__file__)[0].replace("configs", "results", 1)
    ),
    MODEL=models.LLAMA2_7B_CHAT_CONFIG,
    EVAL=dict(
        NUM_EXAMPLES=2,
    )
)
