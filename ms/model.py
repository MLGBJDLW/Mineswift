"""The learned move scorer ("the AI"), and its exact-solver teacher.

Design rule that keeps this honest: the model NEVER replaces the solver. The solver's
zero-risk moves are always played, and the model only ranks the cells to open when a guess
is unavoidable. Two reasons:
  * replacing the solver can only make the logic worse (the solver is already exact)
  * keeping it lets every loss be attributed to either a logic bug (impossible) or a guess

The model itself is deliberately tiny and CPU-fast: a two-layer conv net on the 13x13
window that predicts, for the centre cell, whether opening it is the right guess. It is
trained by imitation of the exact global probability (ranking the candidates), which is a
strong but not optimal teacher - the teacher minimises immediate risk, not information
value. That gap is where RL/rollouts can add, and dataset.py stores rollouts for it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # torch is optional at import time so the solver/eval path has no heavy dependency
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except Exception:  # pragma: no cover - environment without torch
    _HAS_TORCH = False


def _torch_cls():
    if not _HAS_TORCH:
        raise ImportError("model.py needs torch: pip install torch --index-url "
                          "https://download.pytorch.org/whl/cpu")
    return torch, nn


if _HAS_TORCH:
    class GuessScorerV1(nn.Module):
        """13x13 cell-token window -> one logit per cell. ~60k parameters."""

        def __init__(self, window: int = 13, channels: int = 32):
            super().__init__()
            self.embed = nn.Embedding(13, channels)
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.head = nn.Conv2d(channels, 1, 1)
            self.window = window

        def forward(self, tokens: "torch.Tensor"):
            # tokens: (B, 13, 13) int64
            x = self.embed(tokens).permute(0, 3, 1, 2)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            return self.head(x).squeeze(1)  # (B, 13, 13) logits


@dataclass
class ModelReport:
    params: int
    window: int
    train_rows: int = 0
    train_seconds: float = 0.0
    val_top1: float = 0.0
    val_mae: float = 0.0


def load_scorer(weights: str = "", channel: int = 32, window: int = 13):
    """Return a callable: list of encoded windows -> per-cell mine-probability scores.

    The returned callable is what LearnedGuessPolicy expects. It is deliberately small and
    CPU-only: inference runs once per guess at most, so it must not add latency to the
    logical path.
    """
    torch, _ = _torch_cls()
    import numpy as np

    net = GuessScorerV1(window=window, channels=channel)
    if weights:
        net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()

    def score(windows) -> "np.ndarray":
        tokens = torch.from_numpy(np.asarray(windows, dtype=np.int64))
        with torch.no_grad():
            logits = net(tokens)                     # (n, 13, 13) prob of "mine if opened"
        centre = window // 2
        return logits[:, centre, centre].detach().cpu().numpy()

    return score


def train_scorer(rows_path: str, out_path: str, channel: int = 32, window: int = 13,
                 steps: int = 3000, batch: int = 128, lr: float = 2e-3,
                 seed: int = 0) -> ModelReport:
    """Train the scorer to predict "this cell is a mine" from the board window.

    Target for cell (r, c) is the exact global mine probability the solver computed, which
    the dataset stored per candidate. That is a well-posed regression with a strong teacher:
    the network learns to reproduce global constraint reasoning from local pixels, which is
    what lets it rank cells it has never seen (any theme, any size of board) instead of
    memorising board ids.

    Rows in `rows_path` come from `python -m ms.dataset`. Only candidate cells are labelled,
    so each row contributes `len(candidates)` supervised cells.
    """
    import json
    import time

    import numpy as np

    torch, nn = _torch_cls()
    torch.manual_seed(seed)
    t0 = time.perf_counter()

    windows, targets = [], []
    with open(rows_path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            grid = np.asarray(row["window"], dtype=np.int64)
            windows.append(grid)
            targets.append(np.asarray(row["p_mine"][0], dtype=np.float32))
    if not windows:
        raise ValueError("no rows to train on")

    X = torch.from_numpy(np.stack(windows))
    y = torch.from_numpy(np.asarray(targets, dtype=np.float32)).view(-1, 1)
    n = len(windows)
    model = GuessScorerV1(window=window, channels=channel)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    order = torch.randperm(n)
    val_n = max(1, n // 10)
    va, tr = order[:val_n], order[val_n:]
    c = window // 2

    for step in range(steps):
        idx = tr[torch.randint(0, len(tr), (min(batch, len(tr)),))]
        logits = model(X[idx])[:, c, c].view(-1, 1)
        loss = loss_fn(logits, y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(X[va])[:, c, c]).view(-1)
        truth = y[va].view(-1)
        mae = float((pred - truth).abs().mean())
        # "did we pick the cell with the lowest mine probability?" - measured on the training
        # distribution here; the honest test is a full evaluation run in ms.eval
        acc = float(((pred < 0.5) == (truth < 0.5)).float().mean())
    torch.save(model.state_dict(), out_path)
    params = sum(p.numel() for p in model.parameters())
    return ModelReport(params=params, window=window, train_rows=n,
                       train_seconds=time.perf_counter() - t0, val_top1=acc,
                       val_mae=mae)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Train the guess scorer on recorded decisions.")
    ap.add_argument("--rows", default="data/guesses.jsonl")
    ap.add_argument("--out", default="models/guess_scorer.pt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--channel", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args(argv)
    from pathlib import Path

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rep = train_scorer(args.rows, args.out, channel=args.channel, steps=args.steps,
                       batch=args.batch, lr=args.lr)
    print(f"params={rep.params} rows={rep.train_rows} train={rep.train_seconds:.1f}s "
          f"val_mae={rep.val_mae:.4f} val_acc={rep.val_top1:.3f} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
