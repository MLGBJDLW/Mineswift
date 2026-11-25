"""Public contract checks for exact posterior sampling on small boards."""
from collections import Counter
from itertools import combinations
from random import Random

import numpy as np
import pytest

from ms.board import COVERED as C, FLAG as F, QUESTION as Q, UNREADABLE as U, Board
from ms.posterior import sample_layouts


def _board(cells, mines_left, frame_id=0):
    visible = np.array(cells, dtype=np.int8)
    return Board(*visible.shape, visible, mines_left, frame_id)


def _brute_force(board):
    """Enumerate whole boards directly, without the sampler or solver constraints."""
    cells = list(np.ndindex(board.cells.shape))
    hidden = [i for i in cells if board.cells[i] in (C, Q)]
    flags = {i for i in cells if board.cells[i] == F}
    clues = [
        (int(board.cells[r, c]),
         {i for i in cells if i != (r, c) and abs(i[0] - r) <= 1 and abs(i[1] - c) <= 1})
        for r, c in cells if board.cells[r, c] >= 0
    ]
    possible = set()
    for extra in combinations(hidden, board.mines_left):
        mines = flags.union(extra)
        if all(len(neighbors & mines) == digit for digit, neighbors in clues):
            possible.add(tuple(i in mines for i in cells))
    return possible


def _assert_samples(samples, board, draws):
    assert isinstance(samples, list)
    assert len(samples) == draws
    for layout in samples:
        assert isinstance(layout, np.ndarray)
        assert layout.shape == (board.rows * board.cols,)
        assert layout.dtype == np.dtype(bool)
    return [tuple(layout) for layout in samples]


def _two_components():
    # Each C-1-C-1-C group permits either one central mine or both end mines.
    # The three unconstrained cells in between make the numbers of completions
    # for group mine-count pairs (1,1), (1,2), (2,1), (2,2) equal to 3, 3, 3, 1.
    return _board([[C, 1, C, 1, C, C, C, C, C, 1, C, 1, C]], 4)


def test_uniform_full_layouts_across_components_and_free_cells():
    board = _two_components()
    expected = _brute_force(board)
    assert len(expected) == 10

    draws = 6000
    samples = _assert_samples(sample_layouts(board, draws, Random(3521)), board, draws)
    counts = Counter(samples)
    assert counts.keys() == expected
    target = draws / len(expected)
    for count in counts.values():
        assert count == pytest.approx(target, rel=0.20)

    # Non-overlapping pairs should repeat a full layout at rate 1 / |expected|.
    # A cycling or batched-without-replacement sampler can get frequencies right.
    repeats = sum(a == b for a, b in zip(samples[::2], samples[1::2]))
    assert repeats / (draws // 2) == pytest.approx(1 / len(expected), abs=0.03)


def test_flags_are_mines_and_questions_are_unresolved():
    board = _board([[F, Q, C], [2, C, C]], 2)
    expected = _brute_force(board)
    assert len(expected) == 4
    samples = _assert_samples(sample_layouts(board, 128, Random(918)), board, 128)
    assert set(samples) == expected
    assert all(layout[0] and sum(layout) == 3 for layout in samples)
    assert {layout[1] for layout in samples} == {False, True}


def test_same_seed_replays_without_mutating_board():
    board = _board([[F, Q, C], [2, C, C]], 2, frame_id=41)
    before = board.cells.copy()
    another_frame = _board(before, 2, frame_id=42)
    first = _assert_samples(sample_layouts(board, 128, Random(8721)), board, 128)
    np.testing.assert_array_equal(board.cells, before)
    second = _assert_samples(sample_layouts(another_frame, 128, Random(8721)), board, 128)
    np.testing.assert_array_equal(another_frame.cells, before)
    assert (board.mines_left, board.frame_id) == (2, 41)
    assert first == second


@pytest.mark.parametrize(("cells", "mines_left"), [
    pytest.param([[U, C]], 0, id="unreadable"),
    pytest.param([[C]], -1, id="negative-mine-count"),
    pytest.param([[C]], 2, id="too-many-mines"),
    pytest.param([[F, 0]], 0, id="contradictory-clue-without-hidden-neighbors"),
    pytest.param([[0, C, 1]], 1, id="incompatible-clues"),
    pytest.param([[C, 1, C]], 0, id="locally-possible-but-global-count-impossible"),
])
def test_unreadable_or_inconsistent_board_has_no_samples(cells, mines_left):
    board = _board(cells, mines_left)
    before = board.cells.copy()
    assert sample_layouts(board, 4, Random(7)) is None
    np.testing.assert_array_equal(board.cells, before)


def test_layout_limit_is_per_component_not_per_whole_board():
    board = _two_components()
    assert sample_layouts(board, 4, Random(7), max_component_layouts=1) is None
    samples = sample_layouts(board, 32, Random(7), max_component_layouts=2)
    assert set(_assert_samples(samples, board, 32)) <= _brute_force(board)


def test_node_limit_is_per_component_and_exhaustion_returns_none():
    one = _board([[C, 1, C, 1, C]], 2)
    # Infer the smallest successful budget through the public API, so the test
    # does not depend on the number of nodes used by a particular enumerator.
    assert sample_layouts(one, 1, Random(7), max_nodes=0) is None
    lo, hi = 0, 6000
    assert sample_layouts(one, 1, Random(7), max_nodes=hi) is not None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if sample_layouts(one, 1, Random(7), max_nodes=mid) is None:
            lo = mid
        else:
            hi = mid
    two = _two_components()
    assert sample_layouts(two, 4, Random(7), max_nodes=hi) is not None
