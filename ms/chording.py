"""Compress one solver `open` into the fewest mouse actions.

Chording = pressing both buttons (or the middle button) on an already-revealed number.
When the flags around that number equal its value, every other covered neighbour opens
in one action. Otherwise the click does nothing.

Why this is an execution concern and not a solver concern:

    A chord-legal number means  number - flags == 0, so constraint propagation already
    produces a Constraint(uncovered_neighbours, 0) and the solver already returns those
    cells as provably safe. The information is identical; only the number of mouse
    actions differs. Therefore chording cannot change a game's outcome - it can only
    change how long the game takes in wall-clock time.

Corollary: chording is only safe when every flag is trustworthy. In this project flags
come from `p == 1.0` deductions, so they are. Flags placed by a human are the one case
that can make a chord explode, which is why `Agent` must not chord on a board it did not
flag itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from .board import COVERED, FLAG, QUESTION, Board

NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

MIN_CHORD_CELLS = 2  # a chord that opens one cell is just a click with extra risk


@dataclass(frozen=True)
class Action:
    """One mouse action the executor must perform."""

    kind: Literal["open", "chord"]
    cell: tuple[int, int]                    # what to click / chord on
    opens: tuple[tuple[int, int], ...]       # cells this action is expected to reveal


def _nbrs(rows: int, cols: int, r: int, c: int):
    for dr, dc in NEIGHBOURS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            yield rr, cc


def chord_candidates(board: Board):
    """Every (number_cell, openable_neighbours) whose flags currently match its value."""
    rows, cols = board.rows, board.cols
    out: list[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = []
    for r in range(rows):
        for c in range(cols):
            v = int(board.cells[r, c])
            if v < 0:
                continue
            nbrs = list(_nbrs(rows, cols, r, c))
            if sum(1 for rr, cc in nbrs if int(board.cells[rr, cc]) == FLAG) != v:
                continue
            opens = tuple(
                (rr, cc) for rr, cc in nbrs
                if int(board.cells[rr, cc]) in (COVERED, QUESTION)
            )
            if opens:
                out.append(((r, c), opens))
    return out


def plan_actions(board: Board, cells: Iterable[tuple[int, int]]) -> list[Action]:
    """Cover `cells` (all proved safe by the solver) with the fewest mouse actions.

    Greedy set cover over the legal chords, then one plain click per leftover. Greedy is
    used deliberately: chords overlap by at most a handful of cells, and a wrong choice
    costs extra clicks, never correctness - every action opens a subset of `cells`.
    """
    target = {(int(r), int(c)) for r, c in cells}
    remaining = set(target)
    actions: list[Action] = []
    candidates = chord_candidates(board)

    while True:
        best: tuple[tuple[int, int], tuple[tuple[int, int], ...]] | None = None
        for cell, opens in candidates:
            gain = tuple(x for x in opens if x in remaining)
            if len(gain) >= MIN_CHORD_CELLS and (best is None or len(gain) > len(best[1])):
                best = (cell, gain)
        if best is None:
            break
        cell, gain = best
        actions.append(Action("chord", cell, gain))
        remaining.difference_update(gain)

    actions.extend(Action("open", cell, (cell,)) for cell in sorted(remaining))
    return actions
