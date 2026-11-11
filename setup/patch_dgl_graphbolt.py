#!/usr/bin/env python
"""Let DGL 2.1.0 import on torch versions it has no graphbolt library for.
Same one-line change as in ~/Documents/thesis_ch8/.venv. Safe to run twice."""
import importlib.util, os, sys

spec = importlib.util.find_spec("dgl")
if spec is None or spec.origin is None:
    sys.exit("dgl is not installed in this environment")
path = os.path.join(os.path.dirname(spec.origin), "graphbolt", "__init__.py")
src = open(path).read()
MARK = "# patched: graphbolt unused by this project"
if MARK in src:
    print(f"already patched: {path}"); sys.exit(0)
old1 = ('        raise FileNotFoundError(\n'
        '            f"Cannot find DGL C++ graphbolt library at {path}"\n'
        '        )\n')
old2 = '        raise ImportError("Cannot load Graphbolt C++ library")\n'
if old1 not in src or old2 not in src:
    sys.exit(f"unexpected contents in {path}; is this DGL 2.1.0? Not patched.")
open(path, "w").write(src.replace(old1, f"        return  {MARK}\n").replace(old2, "        return\n"))
print(f"patched: {path}")
