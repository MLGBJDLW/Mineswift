import numpy as np
import pytest

from ms.board import COVERED as C, FLAG as F, QUESTION as Q, UNREADABLE as U, Board
from ms.deadguess import forced_pair
from ms.policy import DeadGuessPolicy, EndgamePolicy


def board(cells, mines):
    grid = np.array(cells, dtype=np.int8)
    return Board(*grid.shape, grid, mines)


@pytest.mark.parametrize('hidden', [C, Q])
def test_private_one_mine_pair_is_certified(hidden):
    position = board([[C, hidden], [1, 1]], 1)
    assert forced_pair(position) == ((0, 0), (0, 1))


def test_equal_risk_is_not_enough_if_a_future_clue_distinguishes_it():
    # Middle digit proves exactly one of the two immediately neighboring unknowns
    # is a mine, but opening either outer cell could tell which one.
    position = board([[C, C, 1, C, C]], 2)
    assert forced_pair(position) is None


def test_flagged_distinguishing_neighbors_do_not_reveal_clues():
    position = board([[C, C, F], [1, 2, F]], 1)
    assert forced_pair(position) == ((0, 0), (0, 1))
    assert forced_pair(board([[C, C, U], [1, 2, F]], 1)) is None


def test_preserve_exact_endgame_and_known_safe_moves(monkeypatch):
    monkeypatch.setattr('ms.policy.forced_pair', lambda *_: pytest.fail('must not intervene'))
    for position in [board([[0, C]], 0), board([[C, C], [1, 1]], 1)]:
        assert DeadGuessPolicy().act(position) == EndgamePolicy().act(position)


def test_policy_may_prioritize_certified_pair_when_endgame_search_cannot_solve(monkeypatch):
    # At (0,0)/(0,1) a private 50/50; a dense flag column isolates four free cells
    # to its right. Those cells are individually less risky (one mine among four).
    position = board([[C, C, F, C, C], [1, 3, F, C, C]], 2)
    policy = DeadGuessPolicy()
    monkeypatch.setattr(policy, '_search', lambda *_: None)
    assert policy.solver.decide(position).p_mine == .25
    move = policy.act(position)
    assert move.kind == 'guess' and move.cells == ((0, 0),) and move.p_mine == .5
    assert policy.solver.last_stats['deadguess_switch']
