"""Guard the shared env and benchmark module boundaries."""

import importlib


def test_shared_env_has_legacy_wrapper():
    common_env = importlib.import_module("common.env")
    legacy_env = importlib.import_module("Search_2D.env")

    assert legacy_env.Env is common_env.Env

    env = common_env.Env()
    assert env.x_range == 51
    assert env.y_range == 31
    assert (0, 0) in env.obs


def test_metrics_has_legacy_wrapper():
    benchmarks_metrics = importlib.import_module("benchmarks.metrics")
    legacy_metrics = importlib.import_module("metrics")

    assert legacy_metrics.now_ms is benchmarks_metrics.now_ms
    assert legacy_metrics.describe_path_length([(0, 0), (3, 4)]) == "5.000"
