import importlib.util


def dynamic_import(config_path):
    spec = importlib.util.spec_from_file_location("", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
