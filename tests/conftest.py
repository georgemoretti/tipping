import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="run tests that need the local vLLM server",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-gpu"):
        return
    skip_gpu = pytest.mark.skip(reason="needs vLLM server; use --run-gpu")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
