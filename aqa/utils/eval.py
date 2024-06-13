from copy import deepcopy


def eval(config):
    # to avoid circular imports
    from aqa.benchmarks import build_benchmark
    from aqa.models import build_model

    benchmark = build_benchmark(config)
    model = build_model(config)

    eval_config = deepcopy(config.EVAL)
    eval_config = dict({k.lower(): v for k, v in eval_config.items()})

    metric, full_result = benchmark.test_with_examples(
        model, **eval_config
    )

    return metric, full_result
