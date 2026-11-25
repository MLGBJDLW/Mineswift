"""Near-risk midgame choices from exact conditional mine counts, not hidden layouts."""
from __future__ import annotations

from dataclasses import dataclass

from .board import COVERED, FLAG, QUESTION, UNREADABLE, Board
from .solver import BudgetExceeded, Cell, ComponentCache, Inconsistent, Solver, _nbrs


@dataclass(frozen=True)
class GuessValue:
    two_click_survivors: int
    progress_worlds: int
    safe_cell_mass: int

    @property
    def rank(self) -> tuple[int, int, int]:
        # Prefer two-click survival, with a modest bonus for getting a certain open.
        # All candidates share the same denominator (number of original layouts).
        return (10 * self.two_click_survivors + self.progress_worlds,
                self.two_click_survivors, self.safe_cell_mass)


def conditional_value(board: Board, cell: Cell, survive: int, solver: Solver,
                      cache: ComponentCache) -> GuessValue | None:
    """Sum over safe digit observations and take the best next open in each branch.

    The second observation is not simulated. A zero is scored from its digit alone
    (a conservative proxy, rather than assuming unseen flood digits). A completed
    game needs no second click. Return no value if the partition is incomplete.
    """
    neighbors = [int(board.cells[rc]) for rc in _nbrs(board.rows, board.cols, *cell)]
    flags = neighbors.count(FLAG)
    hidden = neighbors.count(COVERED) + neighbors.count(QUESTION)
    cells = board.cells.copy()
    branch = Board(board.rows, board.cols, cells, board.mines_left, board.frame_id)
    seen = second = progress = clear_mass = 0
    for number in range(flags, flags + min(hidden, board.mines_left) + 1):
        cells[cell] = number
        counts: dict[Cell, int] = {}
        stats: dict = {}
        try:
            solver.measure(branch, stats, exact_only=True, mine_counts=counts,
                           component_cache=cache)
        except Inconsistent:
            continue
        worlds = stats['solutions']
        seen += worlds
        if len(counts) == board.mines_left:
            second += worlds
            progress += worlds
        else:
            second += worlds - min(counts.values())
            clears = sum(value == 0 for value in counts.values())
            if clears:
                progress += worlds
                clear_mass += clears * worlds
    if seen != survive:
        return None
    return GuessValue(second, progress, clear_mass)


class MidgameRanker:
    """Score up to eight candidates; keep the baseline on missing exact evidence."""

    def __init__(self, max_candidates: int = 8, max_nodes: int = 2200):
        self.max_candidates = max_candidates
        self.solver = Solver(max_nodes=max_nodes)

    def shortlist(self, board: Board, baseline: Cell, counts: dict[Cell, int],
                  worlds: int) -> list[Cell]:
        # Permit a loss of at most 4% of the safest move's immediate survival, not
        # four percentage points of mine risk on every position.
        risk = counts[baseline]
        extra = (worlds - risk) * 4 // 100
        remaining = set(counts)
        frontier = {rc for rc in remaining if any(board.cells[n] >= 0
                    for n in _nbrs(board.rows, board.cols, *rc))}

        def profile(cell: Cell) -> tuple[tuple[Cell, ...], int]:
            near = set(_nbrs(board.rows, board.cols, *cell)) & remaining
            return tuple(sorted(near & frontier)), len(near - frontier)

        def order(cell: Cell):
            near_frontier, near_free = profile(cell)
            return counts[cell], -len(near_frontier), -near_free, cell

        options = sorted((rc for rc, count in counts.items()
                          if rc != baseline and count <= risk + extra), key=order)
        # Far-off frontier cells with the same adjacency to constrained cells and
        # the same number of exchangeable free neighbors give identical digit scores.
        seen_free = {profile(baseline)} if baseline not in frontier else set()
        equal, close = [], []
        for cell in options:
            if cell not in frontier:
                key = profile(cell)
                if key in seen_free:
                    continue
                seen_free.add(key)
            (equal if counts[cell] == risk else close).append(cell)
        # A large set of exactly tied cells must not crowd out near-risk alternatives.
        slots = self.max_candidates - 1
        reserve = min(len(close), slots // 2)
        same = equal[:slots - reserve]
        return [baseline, *same, *close[:slots - len(same)]]

    def choose(self, board: Board, baseline: Cell, counts: dict[Cell, int],
               worlds: int) -> tuple[Cell, GuessValue] | None:
        if (board.cells == UNREADABLE).any():
            return None
        shortlist = self.shortlist(board, baseline, counts, worlds)
        if len(shortlist) < 2:
            return None
        cache: ComponentCache = {}
        best: tuple[Cell, GuessValue] | None = None
        for cell in shortlist:
            survive = worlds - counts[cell]
            if best is not None and 11 * survive < best[1].rank[0]:
                continue  # even perfect subsequent play cannot beat the incumbent
            try:
                value = conditional_value(board, cell, survive, self.solver, cache)
            except BudgetExceeded:
                value = None
            if value is None:
                if cell == baseline:
                    return None  # no evidence that switching improves over the reference
                continue
            if best is None or value.rank > best[1].rank:
                best = cell, value
        return best
