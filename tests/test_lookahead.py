"""The lookahead may change information value, never the mine risk of a guess."""
from itertools import combinations

import numpy as np
import pytest

from ms.board import COVERED as C, FLAG as F, QUESTION as Q, Board
from ms.lookahead import OneStepRanker, _reveal_score
from ms.policy import ExactSolverPolicy, LookaheadPolicy
from ms.solver import Inconsistent, Solver, enumerate_group


def _board(cells, mines_left):
    arr = np.array(cells, dtype=np.int8)
    return Board(*arr.shape, arr, mines_left)


def _neighbors(board, cell):
    return {rc for rc in np.ndindex(board.cells.shape)
            if 0 < max(abs(a - b) for a, b in zip(rc, cell)) <= 1}


@pytest.mark.parametrize("board", [
    _board([[C, C, C, C, C]], 1),
    _board([[C, 1, C, Q, C, C]], 2),
    _board([[F, Q, C], [2, C, C], [C, C, C]], 2),
])
def test_reveal_scores_match_brute_force(board):
    hidden = {rc for rc in np.ndindex(board.cells.shape) if board.cells[rc] in (C, Q)}
    flags = set(zip(*np.where(board.flags())))
    clues = {rc: _neighbors(board, rc) for rc in np.ndindex(board.cells.shape)
             if board.cells[rc] >= 0}
    worlds = [set(pick) for pick in combinations(sorted(hidden), board.mines_left)
              if all(len(nb & (set(pick) | flags)) == board.cells[rc] for rc, nb in clues.items())]
    assert worlds
    stats, counts = {}, {}
    solver = Solver()
    solver.measure(board, stats, mine_counts=counts, exact_only=True)
    assert stats["solutions"] == len(worlds)
    assert counts == {rc: sum(rc in world for world in worlds) for rc in hidden}
    original = board.cells.copy()
    for cell in hidden:
        buckets = {}
        nb = _neighbors(board, cell)
        for world in worlds:
            if cell not in world:
                buckets.setdefault(len(nb & (world | flags)), []).append(world)
        progress = total = 0
        for bucket in buckets.values():
            safe = len(hidden - {cell} - set.union(*bucket))
            if safe:
                progress += len(bucket)
                total += safe * len(bucket)
        assert _reveal_score(board, cell, len(worlds) - counts[cell], solver) == (progress, total)
    np.testing.assert_array_equal(board.cells, original)


def test_tie_prefers_guess_that_proves_more_safe_cells():
    board = _board([[C, C, C, C, C]], 1)
    base = ExactSolverPolicy().act(board)
    alternative = LookaheadPolicy().act(board)
    assert base.kind == alternative.kind == "guess"
    assert base.cells == ((0, 0),) and alternative.cells == ((0, 1),)
    assert base.p_mine == alternative.p_mine == 0.2
    assert LookaheadPolicy().act(board) == alternative


@pytest.mark.parametrize("board", [_board([[0, C]], 0), _board([[1, C]], 1),
                                  _board([[C, 1, C]], 1), _board([[0]], 0)])
def test_no_switch_on_deductions_or_symmetric_guess(board):
    assert LookaheadPolicy().act(board) == ExactSolverPolicy().act(board)


def test_ranker_cannot_increase_the_mine_risk():
    board = _board([[C, 1, C, C, C, C]], 2)
    solver = Solver()
    base = solver.decide(board)
    assert base.cells[0] not in ((0, 0), (0, 2))
    assert solver.decide(board, guess_ranker=lambda *_: (0, 0)) == base


def test_approximate_measurement_never_calls_the_ranker(monkeypatch):
    board = _board([[C, 1, C, C, C]], 2)
    monkeypatch.setattr("ms.solver.sample_group", lambda g, *_: enumerate_group(g, 100))
    solver = Solver(max_nodes=0)

    def fail(*_):
        pytest.fail("sampling cannot establish a lookahead tie")

    assert solver.decide(board, guess_ranker=fail).kind == "guess"
    assert solver.last_stats["degraded"]


def test_branch_budget_returns_original_move():
    board = _board([[C, C, C, C, C]], 2)
    solver = Solver()
    # Some hypothetical digit-1 outcomes require more than one DFS node.
    assert solver.decide(board, guess_ranker=OneStepRanker(max_nodes=1)) == solver.decide(board)


@pytest.mark.parametrize("board", [_board([[1, 0]], 0), _board([[0, C, 2, C]], 2)])
def test_impossible_hypothetical_clues_are_rejected(board):
    with pytest.raises(Inconsistent):
        Solver().measure(board, exact_only=True)
