"""Exact whole-game win probability for bounded Minesweeper endgames."""
from __future__ import annotations

from dataclasses import dataclass

from .board import Board, UNREADABLE
from .engine import neighbour_ids
from .solver import BudgetExceeded, Cell, Constraint, Inconsistent, enumerate_group, raw_constraints


@dataclass(frozen=True)
class EndgameResult:
    cell: Cell
    winning_worlds: int
    worlds: int
    mine_worlds: int

    @property
    def p_win(self) -> float:
        return self.winning_worlds / self.worlds


def _indices(bits: int):
    while bits:
        low = bits & -bits
        yield low.bit_length() - 1
        bits ^= low


def _worlds(board: Board, hidden: tuple[int, ...], expected: int, max_nodes: int) -> list[int]:
    clues, _, _ = raw_constraints(board, strict=True)
    if not 0 <= board.mines_left <= len(hidden):
        raise Inconsistent("endgame mine budget is impossible")
    worlds: list[int] = []

    def collect(mask: int) -> None:
        if len(worlds) == expected:
            raise Inconsistent("endgame enumeration exceeded the counted layouts")
        worlds.append(mask)

    # Coupling the components with the global count also includes all unbordered cells.
    enumerate_group([*clues, Constraint(hidden, board.mines_left)], max_nodes,
                    on_solution=collect)
    if len(worlds) != expected:
        raise Inconsistent("endgame enumeration did not match the counted layouts")
    return worlds


def _observations(board: Board, hidden: tuple[int, ...], worlds: list[int]):
    """Per layout and click: (flood mask, revealed digits encoded on four bit planes)."""
    local = {cell: i for i, cell in enumerate(hidden)}
    flags = board.flags().ravel()
    nb = neighbour_ids(board.rows, board.cols)
    neighborhood = [[int(j) for j in nb[cell] if j >= 0] for cell in hidden]
    masks = [sum(1 << local[j] for j in ns if j in local) for ns in neighborhood]
    known_mines = [sum(bool(flags[j]) for j in ns) for ns in neighborhood]
    rows: list[list[tuple[int, tuple[int, ...]]]] = []
    for mines in worlds:
        digits = [(mask & mines).bit_count() + marked for mask, marked in zip(masks, known_mines)]
        planes = [sum(1 << i for i, digit in enumerate(digits) if digit & (1 << b))
                  for b in range(4)]
        zeros = sum(1 << i for i, digit in enumerate(digits) if digit == 0)
        row = []
        for i, digit in enumerate(digits):
            flood = 1 << i
            if not (flood & mines) and digit == 0:
                pending = flood
                while pending:
                    z = pending & -pending
                    pending ^= z
                    new = masks[z.bit_length() - 1] & ~flood
                    flood |= new
                    pending |= new & zeros
            row.append((flood, tuple(p & flood for p in planes)))
        rows.append(row)
    return rows


class _Search:
    def __init__(self, worlds: list[int], rows, mines_left: int, cells: int, max_work: int):
        self.rows = rows
        self.mines_left = mines_left
        self.max_work = max_work
        self.mine_worlds = [sum(1 << w for w, mines in enumerate(worlds) if mines & (1 << cell))
                            for cell in range(cells)]
        self.memo: dict[tuple[int, int], int] = {}

    def _spend(self, amount: int) -> None:
        self.max_work -= amount
        if self.max_work < 0:
            raise BudgetExceeded

    def _click(self, remaining: int, eligible: int, cell: int) -> int:
        """Sum winning worlds across *observable* branches, not per-world best moves."""
        self._spend(eligible.bit_count())
        groups: dict[tuple[int, ...], int] = {}
        for world in _indices(eligible):
            flood, digits = self.rows[world][cell]
            # Previously revealed zeros can be revisited harmlessly: all their adjacent
            # cells were opened by that earlier flood. Only new observations matter.
            key = (remaining & ~flood, *(p & remaining for p in digits))
            groups[key] = groups.get(key, 0) | (1 << world)
        return sum(self.value(key[0], group) for key, group in groups.items())

    def value(self, remaining: int, feasible: int) -> int:
        n_worlds = feasible.bit_count()
        # In a single layout we know every safe tile; nothing is left to guess.
        if n_worlds == 1 or remaining.bit_count() == self.mines_left:
            return n_worlds
        key = (remaining, feasible)
        if key not in self.memo:
            self.memo[key] = self.pick(remaining, feasible)[1]
        return self.memo[key]

    def pick(self, remaining: int, feasible: int, preferred: int | None = None) -> tuple[int, int]:
        self._spend(remaining.bit_count())
        candidates = []
        for i in _indices(remaining):
            safe = feasible & ~self.mine_worlds[i]
            if safe == feasible:
                # A free observation cannot reduce the best achievable win probability.
                return i, self._click(remaining, safe, i)
            if safe:
                candidates.append((i, safe, safe.bit_count()))
        candidates.sort(key=lambda entry: (entry[0] != preferred, -entry[2], entry[0]))
        best_cell, best_count = -1, -1
        for cell, safe, survivors in candidates:
            if survivors > best_count:  # survival is an upper bound on eventual wins
                won = self._click(remaining, safe, cell)
                if won > best_count:
                    best_cell, best_count = cell, won
        return best_cell, best_count


def solve_endgame(board: Board, baseline: Cell, expected_worlds: int, *, max_cells: int = 48,
                  max_worlds: int = 160, max_nodes: int = 6000,
                  max_work: int = 100_000) -> EndgameResult | None:
    """Return an optimal action for this bounded visible position, or no recommendation."""
    if not 1 < expected_worlds <= max_worlds:
        return None
    hidden = tuple(int(i) for i in board.covered().ravel().nonzero()[0])
    if not board.mines_left < len(hidden) <= max_cells or (board.cells == UNREADABLE).any():
        return None
    preferred = baseline[0] * board.cols + baseline[1]
    if preferred not in hidden:
        return None
    try:
        worlds = _worlds(board, hidden, expected_worlds, max_nodes)
        search = _Search(worlds, _observations(board, hidden, worlds), board.mines_left,
                         len(hidden), max_work)
        cell, wins = search.pick((1 << len(hidden)) - 1, (1 << len(worlds)) - 1,
                                 hidden.index(preferred))
    except (BudgetExceeded, Inconsistent):
        return None
    if cell < 0:
        return None
    return EndgameResult(divmod(hidden[cell], board.cols), wins, len(worlds),
                         search.mine_worlds[cell].bit_count())
