#!/usr/bin/env python
"""Check the environment before running any notebook.

1. torch can launch a real kernel on the GPU (a cu118 build on a 5090 passes
   torch.cuda.is_available() and then fails on the first op).
2. DGL imports and can build a line graph and sample blocks on the CPU.
3. ddi.model.ScatterSAGEConv runs on the GPU on a CPU-sampled block and gives
   the same output as dgl.nn.SAGEConv (checked on the CPU).
"""
import sys


def main():
    import torch
    print("python:", sys.version.split()[0], "| torch:", torch.__version__)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        print("device:", torch.cuda.get_device_name(0),
              "| capability: sm_%d%d" % torch.cuda.get_device_capability(0))
        try:
            a = torch.randn(2048, 2048, device="cuda")
            print("GPU matmul OK, checksum:", round(float((a @ a).sum()), 2))
        except Exception as e:
            print("FAIL: kernel launch failed ->", e)
            print("Wrong CUDA build for this GPU (5090 needs cu128).")
            sys.exit(1)
    else:
        print("WARNING: no GPU visible; continuing on CPU")

    import dgl
    from dgl.dataloading import DataLoader, NeighborSampler
    from dgl.nn import SAGEConv
    print("dgl:", dgl.__version__)
    from ddi.model import ScatterSAGEConv

    g = dgl.rand_graph(300, 3000)
    lg = dgl.line_graph(g, backtracking=False)
    lg.ndata["x"] = torch.randn(lg.num_nodes(), 16)
    loader = DataLoader(lg, torch.arange(64), NeighborSampler([5]), batch_size=64)
    _, _, blocks = next(iter(loader))
    x = blocks[0].srcdata["x"]

    torch.manual_seed(0)
    ref = SAGEConv(16, 8, aggregator_type="mean")
    ours = ScatterSAGEConv(16, 8)
    ours.load_state_dict(ref.state_dict())          # same parameter names
    with torch.no_grad():
        diff = (ref(blocks[0], x) - ours(blocks[0], x)).abs().max().item()
    print(f"ScatterSAGEConv vs dgl SAGEConv, max abs diff (CPU): {diff:.2e}")
    assert diff < 1e-5, "ScatterSAGEConv does not match DGL"

    ours = ours.to(dev)
    with torch.no_grad():
        out = ours(blocks[0], x.to(dev))            # block stays on CPU
    print(f"ScatterSAGEConv on {dev}: output {tuple(out.shape)}")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
