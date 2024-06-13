from aqa.configs import Config

BENCHMARK_BASE_CONFIG = Config(
    FORMAT_TOLERANT=True,
    MAX_RETRY=3,
    MAX_STEP=20,
    VERBOSE=True,
    SAVE_PERIOD=-1,
    EXP_NAME=None,
    OUTPUT_DIR=None
)
