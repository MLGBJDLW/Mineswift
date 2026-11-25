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


@pytest.mark.parametrize("board", [
    _board([[C, C, C, C, C, C]], 2), _board([[C, 1, C]], 1),
    _board([[C, C, C], [C, C, C], [C, C, C]], 2),
    _board([[F, Q, C], [2, C, C], [C, C, C]], 2), RISK_TRADEOFF,
])
def test_layout_partitions_match_every_oracle_first_click(board):
    from ms.endgame import _worlds
    from ms.world_partitions import WorldPartitions
    n, oracle = _oracle(board)
    hidden = tuple(map(int, np.flatnonzero(board.covered())))
    worlds = _worlds(board, hidden, n, 10000)
    tree = WorldPartitions(board, hidden, worlds, 1_000_000)
    for i, cell in enumerate(hidden):
        assert tree.click(i, tree.all) == oracle[divmod(cell, board.cols)]
    tree.work = 0
    assert tree.compare(0, range(len(hidden))) is None


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


def test_budgeted_comparison_keeps_only_completed_improvements():
    from ms.verified_search import compare_choices
    from ms.endgame import _worlds
    board = RISK_TRADEOFF
    count, oracle = _oracle(board)
    hidden = tuple(map(int, np.flatnonzero(board.covered())))
    worlds = _worlds(board, hidden, count, 1000)
    order = sorted(range(len(hidden)), key=lambda i: -oracle[divmod(hidden[i], board.cols)])
    assert compare_choices(board, hidden, worlds, 0, order, 0) is None
    cell, original, improved, complete = compare_choices(board, hidden, worlds, 0, order, 50)
    assert not complete
    assert improved == oracle[divmod(hidden[cell], board.cols)] > original == oracle[(0, 0)]
    # Even if another candidate ties first, preserve the incumbent on equality.
    tied = compare_choices(board, hidden, worlds, order[1], order, 10000)
    assert tied == (order[1], improved, improved, True)


def test_verified_guess_displays_risk_and_wins_for_the_selected_cell():
    pytest.importorskip("torch")
    from ms.belief_rl import BeliefActor
    from ms.verified_search import VerifiedActor
    # A fresh uniform actor prefers the first covered tile. This board has
    # a strictly better terminal choice.
    actor = VerifiedActor(BeliefActor().eval(), 0)
    actor.world_limit = 0
    guess = actor.act(RISK_TRADEOFF)
    count, oracle = _oracle(RISK_TRADEOFF)
    assert actor.changes and guess.cells != ((0, 0),)
    assert actor.last_guess["mine_risk"] == guess.p_mine
    assert actor.last_guess["verified_worlds"] == count
    assert actor.last_guess["optimal_wins"] == max(oracle.values()) == oracle[guess.cells[0]]
    assert not actor.last_guess["rollout_worlds"]


def test_conditional_swap_requires_every_distinguishing_neighbor_to_be_mined():
    pytest.importorskip("torch")
    from experiments.swap_guesses import possible_swaps, swap_credit
    board = _board([[C, C, C, C]], 2)
    pairs, outside = possible_swaps(board)
    selected = (pairs == (0, 1)).all(1)
    assert selected.sum() == 1
    # The third cell would distinguish 0 from 1 if it were safe.
    assert set(outside[selected][0]) == {-1, 2}
    worlds = np.zeros((6, 4), dtype=bool)
    for i, mines in enumerate(combinations(range(4), 2)):
        worlds[i, list(mines)] = True
    credit = swap_credit(worlds, pairs[selected], outside[selected])
    assert 0 < credit[0] == credit[1] < (2/6)/2
    assert credit[2] == credit[3] == 0
    # If cell 2 has been opened, no compatible pair is interchangeable.
    pairs, _ = possible_swaps(_board([[C, C, 1, C]], 2))
    assert not (pairs == (0, 1)).all(1).any()


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
