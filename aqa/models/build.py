from copy import deepcopy

from aqa.utils import Registry

MODELS = Registry("models")


def build_model(config):
    config = deepcopy(config.MODEL)
    model_cls = MODELS[config.NAME]

    config.pop("NAME")
    config = dict({k.lower(): v for k, v in config.items()})
    model = model_cls(**config)

    return model
