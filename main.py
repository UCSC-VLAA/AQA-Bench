import argparse
from loguru import logger

from aqa.utils import dynamic_import, eval

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    logger.info(args)

    config = dynamic_import(args.config).config
    opts = [(k, v) for k, v in zip(args.opts[::2], args.opts[1::2])]
    config.update_from_list(opts)

    logger.info(config)
    eval(config)
