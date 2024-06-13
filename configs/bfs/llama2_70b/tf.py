from copy import deepcopy
import os.path as osp

from aqa.configs import benchmarks, models
from aqa.configs.base_config import BASE_CONFIG

config = deepcopy(BASE_CONFIG)
config.update(
    BENCHMARK=dict(
        benchmarks.BFS_CONFIG,
        DATASET_FILE="bfs_0.json",
        SAVE_PERIOD=25,
        EXP_NAME=osp.splitext(osp.split(__file__)[1])[0],
        OUTPUT_DIR=osp.split(__file__)[0].replace("configs", "results", 1)
    ),
    MODEL=models.LLAMA2_70B_CONFIG,
    EVAL=dict(
        NUM_EXAMPLES=0,
        TEACHER_FORCING=True,
    )
)
