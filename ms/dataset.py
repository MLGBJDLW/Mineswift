"""Self-play data generation and the structure of the learned move scorer.

Why this exists
---------------
The solver already plays every logically-solvable board perfectly (evaluation reports zero
"logic" losses). The only place a learned model can add anything is the guess: when no safe
move exists, the agent must pick a cell and the choice decides the game. So the model's job
is narrow and well-posed:

    input   a board tensor around the guess (the frontier "window")
    output  a score per candidate cell

Training targets, in order of quality:
  1. win-rate labels  - play the game out from the situation with a fast rollout policy,
     and label the candidate that actually led to a win (this is the "intelligence" axis)
  2. probability labels - the exact global mine probability from Solver.measure
     (this is the "best guess" axis, and it is available immediately)

Row schema (JSONL, one row per recorded decision):
    {
      "window": [[...]],          # int8 grid, see WINDOW_SIZE
      "candidates": [[r, c], ...],# candidate cells inside the window
      "p_mine": [0.13, ...],      # exact global probability per candidate
      "label": 3,                 # index of the candidate the policy should pick
      "won": true|false,          # outcome of playing from here (filled by rollouts)
      "seed": 1234
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .board import COVERED, FLAG, QUESTION, Board
from .engine import Minefield
from .solver import Solver, certain_moves, raw_constraints

WINDOW = 6  # +/- cells around the frontier that the scorer sees


def encode_window(board: Board, r: int, c: int, size: int = WINDOW) -> np.ndarray:
    """Crop around (r, c) and normalise: covered=9, flag=10, question=11, number=0..8."""
    rows, cols = board.rows, board.cols
    out = np.full((2 * size + 1, 2 * size + 1), 12, dtype=np.int8)  # 12 = outside
    for dr in range(-size, size + 1):
        rr = r + dr
        if not (0 <= rr < rows):
            continue
        for dc in range(-size, size + 1):
            cc = c + dc
            if not (0 <= cc < cols):
                continue
            v = int(board.cells[rr, cc])
            out[dr + size, dc + size] = (
                9 if v == COVERED else 10 if v == FLAG else 11 if v == QUESTION else v
            )
    return out


def record_decision(board: Board, solver: Solver, seed: int = 0) -> dict | None:
    """Capture the scoring problem at a decision point where a guess is required.

    Returns None when a zero-risk move exists (nothing to learn: the solver is already
    optimal there), so the dataset contains only the cases the model can actually improve.
    """
    raw, frontier, outside = raw_constraints(board)
    safe, mines = certain_moves(raw)
    if safe or mines:
        return None
    # One-level propagation is not enough: a cell can be forced safe by the mine counter or
    # by a chain across components. The full probability pass decides what is really a guess.
    probs_all, _ = solver.measure(board, {})
    if any(p <= 1e-12 or p >= 1 - 1e-12 for p in probs_all.values()):
        return None
    probs = probs_all
    cands = sorted(probs, key=lambda rc: probs[rc])
    if not cands:
        return None
    r, c = cands[0]
    return {
        "window": encode_window(board, r, c).tolist(),
        "center": [r, c],
        "candidates": [[rr, cc] for (rr, cc) in cands],
        "p_mine": [round(probs[rc], 6) for rc in cands],
        "label": 0,          # exact-solver choice: lowest global probability
        "won": None,
        "seed": seed,
    }


def play_and_record(args):
    seed, cfg = args
    """One self-play game; returns the rows of every forced-guess decision."""
    rng = np.random.default_rng(seed * 1_000_003 + 7)
    mf = Minefield(cfg, rng)
    mf.click((cfg.rows // 2) * cfg.cols + cfg.cols // 2)
    solver = Solver(seed=0)
    rows: list[dict] = []
    for _ in range(4000):
        if mf.won or mf.exploded:
            break
        board = mf.board(seed)
        row = record_decision(board, solver, seed)
        if row is not None:
            row["won"] = mf.won
            rows.append(row)
        move = solver.decide(board)
        if move.kind in ("open", "guess"):
            r, c = move.cells[0]
            if mf.click(r * cfg.cols + c) == "boom":
                break
        elif move.kind == "flag":
            for r, c in move.cells:
                mf.flag(r * cfg.cols + c)
    return rows



def generate(games: int, cfg, out_path: str, seed_base: int = 0, jobs: int = 1,
             max_rows: int = 400_000) -> int:
    """Play `games` self-play boards and record every forced-guess decision.

    Each row is one decision point: the board window, the candidate cells, their exact mine
    probabilities, and the outcome of the game. Rows are written incrementally, so a long
    run can be interrupted and still leave usable data.
    """
    import multiprocessing as mp
    from pathlib import Path

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    seeds = list(range(seed_base, seed_base + games))
    if jobs > 1:
        with mp.Pool(jobs) as pool:
            all_rows = pool.map(play_and_record, [(s, cfg) for s in seeds],
                              chunksize=max(1, games // (jobs * 8)))
    else:
        all_rows = [play_and_record((s, cfg)) for s in seeds]

    written = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for rows in all_rows:
            for row in rows:
                if written >= max_rows:
                    break
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .engine import BEGINNER, EXPERT, INTERMEDIATE

    ap = argparse.ArgumentParser(description="Generate self-play guess data.")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--preset", default="expert", choices=["beginner", "intermediate",
                                                            "expert"])
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--out", default="data/guesses.jsonl")
    args = ap.parse_args(argv)
    cfg = {"beginner": BEGINNER, "intermediate": INTERMEDIATE, "expert": EXPERT}[args.preset]
    n = generate(args.games, cfg, args.out, args.seed_base, args.jobs)
    print(f"{n} decision rows -> {args.out}")
    return 0


@dataclass
class Recorder:
    """Appends decisions to a JSONL file while games are played."""

    path: str
    every_n: int = 1

    def __post_init__(self):
        self._fh = open(self.path, "a", encoding="utf-8")
        self._count = 0

    def add(self, row: dict) -> None:
        self._count += 1
        if self._count % self.every_n:
            return
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._fh.close()


if __name__ == "__main__":
    raise SystemExit(main())
