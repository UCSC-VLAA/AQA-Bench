import os.path as osp
from loguru import logger

from .file import ensure_dir


def setup_logger(output=None):
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = osp.join(output, "log.txt")
        ensure_dir(osp.dirname(filename))
        logger.add(filename)
