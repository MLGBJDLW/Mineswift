"""Detect a provably unavoidable two-cell 50/50 from the visible board."""
from __future__ import annotations

from .board import FLAG, QUESTION, Board
from .solver import Cell, _nbrs, raw_constraints


def forced_pair(board: Board) -> tuple[Cell, Cell] | None:
    """Find an exactly-one-mine pair whose swap no other safe clue could reveal.

    Require a direct clue telling us there is exactly one mine, and identical
    non-flagged neighborhoods outside the pair. A Flag must be a trusted mine.
    Swapping the pair then preserves all other safe digits, including the clues
    we have not revealed yet. This is stronger than equal marginal risks.
    """
    if ((board.cells < QUESTION) | (board.cells > 8)).any():
        return None
    constraints, _, _ = raw_constraints(board)
    pairs = []
    for constraint in constraints:
        if constraint.mines != 1 or len(constraint.cells) != 2:
            continue
        a, b = (divmod(i, board.cols) for i in constraint.cells)
        ns_a, ns_b = (set(_nbrs(board.rows, board.cols, *c)) for c in (a, b))
        difference = (ns_a ^ ns_b) - {a, b}
        if any(board.cells[c] != FLAG for c in difference):
            continue
        # Neighbors outside the pair differ only by flag constants. If either
        # safe click can reveal zero and expand, it is the one with fewer flags.
        pair = tuple(sorted((a, b), key=lambda c: (
            sum(board.cells[n] == FLAG for n in (ns_a if c == a else ns_b)), c)))
        unknown = sum(board.cells[n] < 0 and board.cells[n] != FLAG
                      for n in (ns_a & ns_b) - {a, b})
        pairs.append((-unknown, pair))
    return min(pairs)[1] if pairs else None
