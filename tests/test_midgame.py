"""Check conditional counts against full-layout enumeration on tiny boards."""
from itertools import combinations

import numpy as np
import pytest

from ms.board import COVERED as C, FLAG as F, QUESTION as Q, Board
from ms.midgame import GuessValue, MidgameRanker, conditional_value
from ms.policy import EndgamePolicy, MidgamePolicy
from ms.solver import Solver


def _board(cells, mines):
    a = np.array(cells, dtype=np.int8)
    return Board(*a.shape, a, mines)


def _neighbors(board, cell):
    return {rc for rc in np.ndindex(board.cells.shape)
            if 0 < max(abs(r - c) for r, c in zip(rc, cell)) <= 1}


@pytest.mark.parametrize('board', [
    _board([[C, C, C, C, C, C]], 2),
    _board([[C, 1, Q, C, C, C]], 2),
    _board([[F, Q, C], [2, C, C], [C, C, C]], 2),
    _board([[C, C]], 1),
])
def test_two_click_value_matches_full_layouts(board):
    hidden = set(zip(*np.where(board.covered())))
    flags = set(zip(*np.where(board.flags())))
    clues = {rc: _neighbors(board, rc) for rc in np.ndindex(board.cells.shape)
             if board.cells[rc] >= 0}
    worlds = [set(try_mines) for try_mines in combinations(sorted(hidden), board.mines_left)
              if all(len((set(try_mines) | flags) & nb) == board.cells[rc]
                     for rc, nb in clues.items())]
    before = board.cells.copy()
    assert worlds
    solver, cache = Solver(), {}
    for click in hidden:
        branches = {}
        for layout in worlds:
            if click not in layout:
                digit = len((layout | flags) & _neighbors(board, click))
                branches.setdefault(digit, []).append(layout)
        options = hidden - {click}
        survives = progress = safe_mass = 0
        for layouts in branches.values():
            if len(options) == board.mines_left:
                survives += len(layouts)
                progress += len(layouts)
                continue
            safe_counts = [sum(rc not in layout for layout in layouts) for rc in options]
            survives += max(safe_counts)
            certainties = safe_counts.count(len(layouts))
            progress += len(layouts) if certainties else 0
            safe_mass += certainties * len(layouts)
        safe_worlds = sum(map(len, branches.values()))
        assert conditional_value(board, click, safe_worlds, solver, cache) == \
            GuessValue(survives, progress, safe_mass)
        assert conditional_value(board, click, safe_worlds + 1, solver, cache) is None
    np.testing.assert_array_equal(before, board.cells)


def test_shortlist_has_risk_guard_and_does_not_consume_real_hidden_state():
    board = _board([[C, 1, C, C, C, C, C]], 2)
    worlds = 200
    counts = {(0, 0): 30, (0, 2): 34, (0, 3): 34, (0, 4): 37, (0, 5): 100, (0, 6): 32}
    shortlist = MidgameRanker().shortlist(board, (0, 0), counts, worlds)
    assert shortlist[0] == (0, 0)
    assert (0, 2) in shortlist
    assert all(counts[c] <= 36 for c in shortlist)


def test_budget_exhaustion_keeps_original():
    ranker = MidgameRanker(max_nodes=0)
    board = _board([[C, C, C, C, C, C]], 2)
    # The middle cell can show 1 with exactly one mine among two neighbors.
    assert ranker.choose(board, (0, 2), {(0, i): 5 for i in range(6)}, 15) is None


def test_keeps_proven_moves_and_exact_endgames(monkeypatch):
    midgame = MidgamePolicy()

    def unexpected(*args):
        pytest.fail('midgame evaluation must not override a deduction or exact endgame')

    monkeypatch.setattr(midgame.ranker, 'choose', unexpected)
    for board in [_board([[0, C]], 0), _board([[C, C, C, C, C, C]], 2)]:
        assert midgame.act(board) == EndgamePolicy().act(board)
