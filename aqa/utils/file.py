import os
import os.path as osp


def ensure_dir(path: str):
    """create directories if *path* does not exist"""""
    if not osp.isdir(path):
        os.makedirs(path, exist_ok=True)
