from collections import Counter

import numpy as np
import pytest

from ms.board import COVERED as C, Board
from ms.policy import EndgamePolicy, RolloutPolicy
from ms.rollout import compare_rollouts
from ms.solver import Move


def board(cells, mines):
    grid = np.array(cells, dtype=np.int8)
    return Board(*grid.shape, grid, mines)


def test_common_worlds_and_terminal_wins(monkeypatch):
    position = board([[C, C, C]], 1)
    # Candidate 0 loses in three worlds; candidate 2 wins those same worlds. In
    # the fourth world both survive their first click, but candidate 2 still wins.
    layouts = [np.array([True, False, False])] * 3 + [np.array([False, True, False])]
    monkeypatch.setattr('ms.rollout.sample_layouts', lambda *_: layouts)
    called = Counter()

    def follow(number):
        called[number] += 1

        def act(observed):
            assert isinstance(observed, Board)
            return Move('guess', ((0, 2),), 0, 'fixture')

        return act

    result = compare_rollouts(position, [(0, 0), (0, 2)], follow, draws=4)
    # In world four, candidate 2 opens 1, then the fixture repeats that same
    # click. Use a useful continuation below to ensure we don't count a loop.
    assert result is None

    def useful(number):
        def act(observed):
            remaining = np.flatnonzero(observed.covered().ravel())
            cell = int(remaining[-1] if observed.cells[0, 2] < 0 else remaining[0])
            return Move('guess', ((0, cell),), 0, 'fixture')
        return act

    result = compare_rollouts(position, [(0, 0), (0, 2)], useful, draws=4)
    assert (result.cell, result.baseline_wins, result.wins, result.challenger_only,
            result.baseline_only) == ((0, 2), 1, 4, 3, 0)
    assert np.array_equal(position.cells, [[C, C, C]])


def test_tie_and_weak_edge_keep_baseline(monkeypatch):
    position = board([[C, C]], 1)
    layouts = [np.array([True, False])] * 2 + [np.array([False, True])]
    monkeypatch.setattr('ms.rollout.sample_layouts', lambda *_: layouts)
    def fail(_):
        return lambda board: pytest.fail('first click is terminal in every sampled world')
    result = compare_rollouts(position, [(0, 0), (0, 1)], fail, draws=3)
    assert result.cell == (0, 0) and result.wins == 1
    assert compare_rollouts(position, [(0, 0), (0, 1)], fail,
                            draws=3, min_edge=1).cell == (0, 1)


def test_rollout_policy_preserves_proven_and_exact_endgame_moves(monkeypatch):
    monkeypatch.setattr('ms.policy.compare_rollouts', lambda *_: pytest.fail('should not rollout'))
    for position in [board([[0, C]], 0), board([[C, C, C, C]], 1)]:
        assert RolloutPolicy().act(position) == EndgamePolicy().act(position)
