"""Small consistency checks for the thesis release."""
import inspect
import pathlib
import re

from ddi import train

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_logged_tail_threshold_is_15():
    p = inspect.signature(train.train_model_transductive).parameters
    assert p["tail_threshold"].default == 15


def test_run_variant_passes_ldam():
    # the thesis settings come from run_variant, not from the defaults
    assert 'loss_type="ldam"' in inspect.getsource(train.run_variant)


def test_test_files_named_in_docstrings_exist():
    named = set()
    for f in (ROOT / "src").rglob("*.py"):
        named |= set(re.findall(r"tests/(test_\w+\.py)", f.read_text()))
    missing = sorted(n for n in named if not (ROOT / "tests" / n).exists())
    assert not missing, f"docstrings name test files that do not exist: {missing}"