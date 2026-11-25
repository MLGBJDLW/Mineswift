"""One-reveal lookahead for exactly tied guesses; no hidden board or random rollouts."""
from __future__ import annotations

from .board import COVERED, FLAG, QUESTION, Board
from .solver import BudgetExceeded, Cell, Inconsistent, Solver, _nbrs


def _reveal_score(board: Board, cell: Cell, safe_worlds: int,
                  solver: Solver) -> tuple[int, int] | None:
    """Count safe branches with a provable next open, then sum their safe-cell counts.

    A branch reveals only the clicked digit. This is deliberately conservative for a 0:
    we count the cells it proves safe, not the information from its full auto-expansion.
    Scores use raw world counts because all compared cells have the same safe denominator.
    """
    neighbors = [int(board.cells[rc]) for rc in _nbrs(board.rows, board.cols, *cell)]
    flags = neighbors.count(FLAG)
    hidden = neighbors.count(COVERED) + neighbors.count(QUESTION)
    cells = board.cells.copy()
    branch = Board(board.rows, board.cols, cells, board.mines_left, board.frame_id)
    seen_worlds = progress_worlds = safe_cell_mass = 0
    for digit in range(flags, flags + min(hidden, board.mines_left) + 1):
        cells[cell] = digit
        stats: dict = {}
        mine_counts: dict[Cell, int] = {}
        try:
            solver.measure(branch, stats, mine_counts=mine_counts, exact_only=True)
        except Inconsistent:
            continue  # impossible digit, not evidence against the candidate
        worlds = stats["solutions"]
        seen_worlds += worlds
        known_safe = sum(count == 0 for count in mine_counts.values())
        if known_safe:
            progress_worlds += worlds
            safe_cell_mass += worlds * known_safe
    # Every possible safe placement must appear in exactly one digit branch.
    if seen_worlds != safe_worlds:
        return None
    return progress_worlds, safe_cell_mass


class OneStepRanker:
    """Bound the extra work by shortlisting at most four exactly tied candidates."""

    def __init__(self, max_candidates: int = 4, max_nodes: int = 1200):
        if max_candidates < 2 or max_nodes < 1:
            raise ValueError("lookahead needs at least two candidates and a positive node budget")
        self.max_candidates = max_candidates
        self._branch_solver = Solver(max_nodes=max_nodes)

    def __call__(self, board: Board, baseline: Cell, mine_counts: dict[Cell, int],
                 worlds: int) -> Cell:
        min_risk = mine_counts[baseline]
        tied = [rc for rc, count in mine_counts.items() if rc != baseline and count == min_risk]
        if not tied:
            return baseline

        def shortlist_order(rc: Cell) -> tuple[int, Cell]:
            # More hidden neighbours can give a wider range of useful observations.
            hidden = sum(board.cells[n] in (COVERED, QUESTION)
                         for n in _nbrs(board.rows, board.cols, *rc))
            return -hidden, rc

        candidates = [baseline, *sorted(tied, key=shortlist_order)[:self.max_candidates - 1]]
        best, best_score = baseline, (-1, -1)
        try:
            for cell in candidates:
                score = _reveal_score(board, cell, worlds - min_risk, self._branch_solver)
                if score is None:
                    return baseline  # do not compare an incomplete branch distribution
                if score > best_score:
                    best, best_score = cell, score
        except BudgetExceeded:
            return baseline  # do not sample or partially rank a shortlist
        return best
