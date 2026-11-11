"""chunk_and_embed: single-chunk SMILES are unchanged, long SMILES average all chunks."""
import torch
from ddi.data import chunk_and_embed


class FakeTokenizer:
    def __call__(self, text, return_tensors="pt"):
        ids = torch.tensor([[ord(c) % 97 for c in text]], dtype=torch.long)
        return {"input_ids": ids}


class FakeOutput:
    def __init__(self, h):
        self.last_hidden_state = h


class FakeModel(torch.nn.Module):
    """[CLS] = a deterministic function of the whole chunk, hidden size 8."""
    def forward(self, input_ids):
        x = input_ids.float()
        cls = torch.stack([x.mean(1), x.std(1, unbiased=False), x.max(1).values,
                           x.min(1).values, x[:, :1].squeeze(1), x[:, -1:].squeeze(1),
                           torch.full_like(x[:, 0], x.shape[1]), x.sum(1) / 1000], dim=1)
        tokens = torch.zeros(x.shape[0], x.shape[1], 8)
        return FakeOutput(torch.cat([cls.unsqueeze(1), tokens], dim=1))


def first_chunk_cls(s, tok, mdl, chunk_size=256):
    return mdl(**tok(s[:chunk_size])).last_hidden_state[:, 0, :]


def test_short_smiles_unchanged():
    tok, mdl = FakeTokenizer(), FakeModel()
    s = "CC(=O)OC1=CC=CC=C1C(=O)O" * 5                    # 120 characters, one chunk
    out = chunk_and_embed(s, tok, mdl)
    assert out.shape == (1, 8)
    assert torch.equal(out, first_chunk_cls(s, tok, mdl))


def test_193_to_256_is_one_chunk():
    """The old chunk rule gave these a redundant second chunk; it must not count."""
    tok, mdl = FakeTokenizer(), FakeModel()
    s = ("C1=CC=CC=C1" * 30)[:200]
    assert torch.equal(chunk_and_embed(s, tok, mdl), first_chunk_cls(s, tok, mdl))


def test_exactly_256_is_one_chunk():
    tok, mdl = FakeTokenizer(), FakeModel()
    s = ("C1=CC=CC=C1" * 30)[:256]
    assert torch.equal(chunk_and_embed(s, tok, mdl), first_chunk_cls(s, tok, mdl))


def test_long_smiles_is_mean_of_all_chunks():
    tok, mdl = FakeTokenizer(), FakeModel()
    s = ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O" * 40)[:700]       # 700 characters
    chunks = [s[0:256], s[192:448], s[384:640], s[576:700]]   # covers all 700 characters
    assert "".join([chunks[0]] + [c[64:] for c in chunks[1:]]) == s
    expected = torch.cat([mdl(**tok(c)).last_hidden_state[:, 0, :] for c in chunks]).mean(0, keepdim=True)
    out = chunk_and_embed(s, tok, mdl)
    assert out.shape == (1, 8)
    assert torch.allclose(out, expected)
    assert not torch.allclose(out, first_chunk_cls(s, tok, mdl))