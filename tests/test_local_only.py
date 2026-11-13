import inspect
import json
import pathlib
import re

import pytest

from ddi import data, paths as P

ROOT = pathlib.Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(r"colab|/content|MyDrive|ON_COLAB|drive\.mount|google\.colab", re.I)
SKIP = {"tests/test_local_only.py", "notebooks/checks/local_only.ipynb",
         "notebooks/checks/db_tests.ipynb"}   # they contain the patterns


def _texts():
    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT).as_posix()
        if (not p.is_file() or rel in SKIP or any(part.startswith(".") for part in p.parts)
                or "__pycache__" in rel or ".egg-info" in rel):
            continue
        if p.suffix == ".ipynb":       # cell sources only, not stored outputs
            cells = json.loads(p.read_text())["cells"]
            yield rel, "\n".join("".join(c["source"]) for c in cells)
        elif p.suffix in {".py", ".sh", ".md", ".toml", ".txt", ".cfg", ".yaml", ".yml"}:
            yield rel, p.read_text()


def test_no_colab_traces():
    hits = [f"{rel}: {m.group(0)!r}" for rel, text in _texts() for m in FORBIDDEN.finditer(text)]
    assert not hits, "Colab traces left:\n" + "\n".join(hits)


def test_colab_setup_script_removed():
    assert not (ROOT / "setup" / "colab_setup.sh").exists()


def test_get_paths_rejects_placeholders_and_expands_home(tmp_path):
    with pytest.raises(ValueError):
        P.get_paths("<DATA_ROOT: e.g. /home/you/ddi_data>", str(tmp_path))
    p = P.get_paths("~/some_data", str(tmp_path))
    assert not p.data_root.startswith("~")
    assert p.drug_db == str(pathlib.Path(p.data_root) / "data" / "Drug_META_DDIE.db")


def test_download_raw_data_takes_paths_and_skips_when_present(tmp_path):
    assert list(inspect.signature(data.download_raw_data).parameters) == ["drug_db", "event_db", "gdrive_id"]
    p = P.get_paths(str(tmp_path / "in"), str(tmp_path / "out"))
    pathlib.Path(p.raw_data_dir).mkdir(parents=True)
    pathlib.Path(p.drug_db).touch()
    pathlib.Path(p.event_db).touch()
    # files present: returns them and never imports gdown or touches the network
    assert data.download_raw_data(p.drug_db, p.event_db, "unused-id") == (p.drug_db, p.event_db)


def test_raw_db_names_only_in_paths():
    for name in ("Drug_META_DDIE.db", "event_META_DDIE_5foldLabel.db"):
        where = sorted(f.relative_to(ROOT).as_posix() for f in (ROOT / "src").rglob("*.py")
                       if name in f.read_text())
        assert where == ["src/ddi/paths.py"], f"{name} hard-coded in {where}"
