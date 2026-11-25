"""Check the endgame tree against brute force using actual simulator observations."""
from functools import lru_cache
from itertools import combinations

import numpy as np
import pytest

from ms.board import COVERED as C, FLAG as F, QUESTION as Q, Board
from ms.endgame import solve_endgame
from ms.engine import Config, Minefield
from ms.policy import DeepEndgamePolicy, EndgamePolicy, ExactSolverPolicy


def _board(cells, budget):
    a = np.array(cells, dtype=np.int8)
    return Board(*a.shape, a, budget)


def _oracle(board):
    """Enumerate full layouts, then replay clicks in fresh Minefields. No solver calls."""
    initial = board.cells.ravel()
    flags = np.flatnonzero(initial == F)
    hidden = np.flatnonzero(board.covered().ravel())
    known = initial >= 0
    cfg = Config(board.rows, board.cols, board.mines_left + len(flags), False)
    rng = np.random.default_rng(0)
    layouts = []
    for mines in combinations(hidden, board.mines_left):
        layout = np.zeros(initial.size, dtype=bool)
        layout[list(mines)] = True
        layout[flags] = True
        mf = Minefield(cfg, rng)
        mf.load(layout)
        if np.array_equal(mf.adjacent[known], initial[known]):
            layouts.append(layout)

    def score(visible, possible, cell):
        n_won, children = 0, {}
        for i in possible:
            mf = Minefield(cfg, rng)
            mf.load(layouts[i])
            mf.visible[:] = np.frombuffer(visible, dtype=np.int8)
            if mf.click(cell) == "boom":
                continue
            if mf.won:
                n_won += 1
            else:
                children.setdefault(mf.visible.tobytes(), []).append(i)
        return n_won + sum(value(state, tuple(ids)) for state, ids in children.items())

    @lru_cache(None)
    def value(visible, possible):
        hidden = np.flatnonzero(np.isin(np.frombuffer(visible, dtype=np.int8), (C, Q)))
        return max(score(visible, possible, int(i)) for i in hidden)

    return (len(layouts), {divmod(int(i), board.cols):
            score(initial.tobytes(), tuple(range(len(layouts))), int(i)) for i in hidden})


RISK_TRADEOFF = _board([[C, 1, 1, C], [2, C, C, 1], [C, C, C, C]], 3)


@pytest.mark.parametrize("board", [
    _board([[C, C, C, C, C, C]], 2),
    _board([[C, 1, C]], 1),
    _board([[F, Q, C], [2, C, C], [C, C, C]], 2),
    RISK_TRADEOFF,
])
def test_matches_simulator_oracle(board):
    original = board.cells.copy()
    worlds, scores = _oracle(board)
    preferred = next(iter(scores))
    answer = solve_endgame(board, preferred, worlds)
    assert answer is not None
    assert answer.worlds == worlds
    assert answer.winning_worlds == scores[answer.cell] == max(scores.values())
    np.testing.assert_array_equal(board.cells, original)


def test_can_choose_lower_survival_for_higher_eventual_win_probability():
    baseline = ExactSolverPolicy().act(RISK_TRADEOFF)
    new = EndgamePolicy().act(RISK_TRADEOFF)
    _, scores = _oracle(RISK_TRADEOFF)
    assert baseline.kind == new.kind == "guess"
    assert new.p_mine == pytest.approx(2 / 7)
    assert new.p_mine > baseline.p_mine
    assert scores[new.cells[0]] == 5 > scores[baseline.cells[0]]


def test_fallback_and_tie_preserve_baseline():
    assert solve_endgame(RISK_TRADEOFF, (1, 2), 7, max_work=0) is None
    assert solve_endgame(RISK_TRADEOFF, (1, 2), 7, max_nodes=0) is None
    assert solve_endgame(RISK_TRADEOFF, (1, 2), 7, max_worlds=6) is None
    assert solve_endgame(RISK_TRADEOFF, (1, 2), 6) is None
    symmetric = _board([[C, 1, C]], 1)
    assert solve_endgame(symmetric, (0, 2), 2).cell == (0, 2)
    assert EndgamePolicy().act(symmetric) == ExactSolverPolicy().act(symmetric)
    certain = _board([[0, C]], 0)
    assert EndgamePolicy().act(certain) == ExactSolverPolicy().act(certain)


def test_deep_policy_searches_past_original_layout_limit():
    # 220 equally plausible layouts exceed the original limit of 160.
    board = _board([[C] * 12], 3)
    standard, deep = EndgamePolicy(), DeepEndgamePolicy()
    first = standard.act(board)
    alternate = deep.act(board)
    assert not standard.solver.last_stats.get("endgame_solved")
    assert deep.solver.last_stats["solutions"] == 220
    assert deep.solver.last_stats["endgame_solved"]
    assert alternate == DeepEndgamePolicy().act(board)
    assert alternate.cells != first.cells
    assert alternate.p_mine == first.p_mine
