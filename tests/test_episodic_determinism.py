"""Episode draws must not depend on Python's string-hash seed, and the
validation macro-F1 is strict over the 175 common classes."""
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn

SNIPPET = r'''
import json, torch
from ddi.experiments import episodic as E
torch.manual_seed(0)
n = 300
labels = torch.cat([torch.arange(1, 13).repeat_interleave(15), torch.arange(176, 188).repeat_interleave(10)])
Z = torch.randn(labels.numel(), 16)
pools = {"common": E.build_label_to_nodes(torch.arange(180), labels),
         "fewer": E.build_label_to_nodes(torch.arange(180, labels.numel()), labels)}
res = E.run_episodic_suite(Z, labels, pools, episodes=20, seed=42, ways=(5,), seed_salt="_t3")
print(json.dumps(E.results_to_json(res), sort_keys=True))
'''


def _run(seed):
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    out = subprocess.run([sys.executable, "-c", SNIPPET], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip().splitlines()[-1]


def test_episodes_independent_of_hash_seed():
    results = {_run(s) for s in (0, 1, 2, 3)}
    assert len(results) == 1, "episode results changed with PYTHONHASHSEED"


def test_salt_is_stable():
    from ddi.experiments.episodic import _salt
    assert [_salt(s) for s in ("common", "fewer", "rare")] == [_salt(s) for s in ("common", "fewer", "rare")]
    assert all(0 <= _salt(s) < 7 for s in ("common", "fewer", "rare", "common_t3"))
    import zlib
    assert _salt("common") == zlib.crc32(b"common") % 7


class _Blk:
    def __init__(self, y):
        self.dstdata = {"label": y}


class _FixedModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def forward(self, blocks):
        return self.logits


def test_validation_macro_f1_is_strict():
    from ddi.experiments import episodic as E
    y = torch.tensor([1, 1, 2, 3])                      # common labels 1..3 only (remapped 0..2)
    logits = torch.zeros(4, E.NUM_COMMON_CLASSES)
    logits[torch.arange(4), torch.tensor([0, 0, 1, 1])] = 1.0   # one error on class 3
    loader = [(None, None, [_Blk(y)])]
    m = E.evaluate(_FixedModel(logits), loader, "cpu")
    per_class = np.array([1.0, 2 / 3, 0.0])            # F1 of classes 1, 2, 3
    assert np.isclose(m["macro_f1"], per_class.sum() / E.NUM_COMMON_CLASSES)   # strict: /175
    assert not np.isclose(m["macro_f1"], per_class.mean())                    # not sklearn default