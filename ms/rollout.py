"""Bounded full-game rollout comparisons from the visible board only."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import blake2b
from random import Random

import numpy as np

from .board import Board
from .engine import Config, Minefield
from .posterior import sample_layouts
from .solver import Cell, Inconsistent, Move


@dataclass(frozen=True)
class RolloutResult:
    cell: Cell
    draws: int
    baseline_wins: int
    wins: int
    challenger_only: int
    baseline_only: int


def _seed(board: Board) -> int:
    digest = blake2b(digest_size=16, person=b'MineRollout-v1')
    digest.update(f'{board.rows},{board.cols},{board.mines_left}:'.encode('ascii'))
    digest.update(board.cells.tobytes())
    return int.from_bytes(digest.digest(), 'big')


def _finish(field: Minefield, index: int, act: Callable[[Board], Move]) -> bool | None:
    status = field.click(index)
    if status in ('boom', 'win'):
        return status == 'win'
    # Each action opens or flags at least one new cell in a legal continuation.
    # The cap also protects against an accidental policy loop.
    for _ in range(field.n):
        try:
            action = act(field.board())
        except Inconsistent:
            return None
        if not action.cells:
            return None
        if action.kind == 'flag':
            for r, c in action.cells:
                field.flag(r * field.cfg.cols + c)
        elif action.kind in ('open', 'guess'):
            # Batch provable opens to reduce overhead; do not batch guesses.
            for r, c in action.cells if action.kind == 'open' else action.cells[:1]:
                status = field.click(r * field.cfg.cols + c)
                if status in ('boom', 'win'):
                    return status == 'win'
        else:
            return None
    return None


def compare_rollouts(board: Board, candidates: Sequence[Cell],
                     continuation: Callable[[int], Callable[[Board], Move]], *,
                     draws: int = 12, min_edge: int = 2) -> RolloutResult | None:
    """Use the same sampled worlds for every action. An incomplete run keeps baseline.

    The continuation sees one visible Board at a time, never the sampled hidden layout.
    All hypotheses are equally weighted. Equal estimates keep candidate zero.
    """
    if len(candidates) < 2 or not 1 <= min_edge <= draws:
        return None
    if len(set(candidates)) != len(candidates) or any(
            not (0 <= r < board.rows and 0 <= c < board.cols and board.covered()[r, c])
            for r, c in candidates):
        raise ValueError('rollout candidates must be distinct covered cells')
    rng = Random(_seed(board))
    layouts = sample_layouts(board, draws, rng)
    if layouts is None:
        return None

    total_mines = int(board.mines_left) + int(board.flags().sum())
    field = Minefield(Config(board.rows, board.cols, total_mines, False),
                      np.random.default_rng(0))  # .load bypasses this unused placement stream
    initial = board.cells.ravel()
    wins = [0] * len(candidates)
    better = [0] * len(candidates)
    worse = [0] * len(candidates)
    for number, layout in enumerate(layouts):
        field.load(layout)
        reference = False
        for j, (r, c) in enumerate(candidates):
            i = r * board.cols + c
            if not layout[i]:
                field.visible[:] = initial
                field.won = field.exploded = False
                won = _finish(field, i, continuation(number))
                if won is None:
                    return None
            else:
                won = False
            wins[j] += won
            if j == 0:
                reference = won
            else:
                better[j] += won and not reference
                worse[j] += reference and not won
    pick = max(range(len(candidates)), key=lambda j: wins[j])
    if wins[pick] - wins[0] < min_edge:
        pick = 0
    return RolloutResult(candidates[pick], draws, wins[0], wins[pick], better[pick], worse[pick])
