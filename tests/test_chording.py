"""Check legal chords and the cells they reveal."""
from __future__ import annotations

import numpy as np
import pytest

from ms.board import COVERED, FLAG, Board
from ms.chording import MIN_CHORD_CELLS, chord_candidates, plan_actions
from ms.engine import EXPERT, Config, Minefield
from ms.solver import Solver


def _board(cells: np.ndarray) -> Board:
    return Board(cells.shape[0], cells.shape[1], cells.astype(np.int8), 1)


def _one_mine_corner() -> tuple[Minefield, Board]:
    """3x3, one mine at (1,1): every cell shows 1, so (0,0) is chord-legal after one flag."""
    mf = Minefield(Config(3, 3, 1), np.random.default_rng(0))
    mines = np.zeros((3, 3), dtype=bool)
    mines[1, 1] = True
    mf.load(mines)
    mf.click(0)  # reveal (0,0); no flood because its count is 1
    return mf, mf.board()


def test_chord_opens_every_unflagged_neighbour():
    mf, _ = _one_mine_corner()
    mf.flag(1 * 3 + 1)
    assert mf.chord(0) == "ok"
    assert int(mf.visible[1]) == 1 and int(mf.visible[3]) == 1
    assert int(mf.visible[4]) == FLAG


def test_chord_needs_exactly_matching_flags():
    mf, _ = _one_mine_corner()
    assert mf.chord(0) == "noop", "no flag: chord must do nothing"
    assert int(mf.visible[1]) == COVERED and int(mf.visible[3]) == COVERED

    mf.flag(1 * 3 + 1)
    mf.flag(0 * 3 + 1)  # a second flag *inside* (0,0)'s neighbourhood
    assert mf.chord(0) == "noop", "flags != number: chord must do nothing"
    assert int(mf.visible[1]) == FLAG and int(mf.visible[3]) == COVERED


def test_a_flag_outside_the_neighbourhood_does_not_count():
    """Regression for a test bug, not a code bug: (2,2) is not a neighbour of (0,0).

    The first version of the test above put the extra flag on (2,2) and expected the
    chord to be refused. On a 3x3 board the corner's neighbourhood is only (0,1), (1,0),
    (1,1), so the chord was - correctly - legal. Pinning both cases keeps that mistake
    from coming back as a "fix" to `chord`.
    """
    mf, _ = _one_mine_corner()
    mf.flag(1 * 3 + 1)
    mf.flag(2 * 3 + 2)
    assert mf.chord(0) == "ok"
    assert int(mf.visible[1]) == 1 and int(mf.visible[3]) == 1


def test_chord_on_a_covered_cell_is_a_noop():
    mf, _ = _one_mine_corner()
    assert mf.chord(1) == "noop"


def test_plan_collapses_a_chordable_pair_into_one_action():
    cells = np.full((3, 3), COVERED, dtype=np.int8)
    cells[0, 0] = 1
    cells[1, 1] = FLAG
    board = _board(cells)

    acts = plan_actions(board, [(0, 1), (1, 0)])
    assert len(acts) == 1 and acts[0].kind == "chord"
    assert acts[0].cell == (0, 0)
    assert set(acts[0].opens) == {(0, 1), (1, 0)}


def test_plan_keeps_single_cells_as_plain_clicks():
    cells = np.full((3, 3), COVERED, dtype=np.int8)
    cells[0, 0] = 1
    cells[1, 1] = FLAG
    board = _board(cells)

    acts = plan_actions(board, [(0, 1)])
    assert [a.kind for a in acts] == ["open"]
    assert len(acts) == 1


def test_plan_opens_each_target_exactly_once():
    cells = np.full((3, 3), COVERED, dtype=np.int8)
    cells[0, 0] = 1
    cells[1, 1] = FLAG
    board = _board(cells)

    target = [(0, 1), (1, 0)]
    acts = plan_actions(board, target)
    opened = [c for a in acts for c in a.opens]
    assert sorted(opened) == sorted(target)
    assert len(opened) == len(set(opened))


def test_candidates_only_report_legal_chords():
    cells = np.full((3, 3), COVERED, dtype=np.int8)
    cells[1, 1] = 2          # value 2
    cells[0, 0] = FLAG       # only one flag
    board = _board(cells)
    assert chord_candidates(board) == []

    cells[0, 1] = FLAG       # now two flags
    board = _board(cells)
    assert [cell for cell, _ in chord_candidates(board)] == [(1, 1)]


# --------------------------------------------------------------------------- #
# end-to-end: same seed, with and without chording
# --------------------------------------------------------------------------- #
def _play(seed: int, use_chords: bool) -> Minefield:
    cfg = EXPERT
    mf = Minefield(cfg, np.random.default_rng(seed * 1_000_003 + 7))
    solver = Solver(seed=seed)

    first = (cfg.rows // 2) * cfg.cols + cfg.cols // 2
    mf.click(first)
    for _ in range(5000):
        if mf.won or mf.exploded:
            break
        board = mf.board()
        move = solver.decide(board)
        if move.kind == "done":
            break
        if move.kind == "flag":
            for r, c in move.cells:
                mf.flag(r * cfg.cols + c)
            continue
        if use_chords and move.kind == "open":
            for act in plan_actions(board, move.cells):
                r, c = act.cell
                if act.kind == "chord":
                    if mf.chord(r * cfg.cols + c) == "boom":
                        raise AssertionError(f"seed {seed}: a legal chord exploded")
                else:
                    mf.click(r * cfg.cols + c)
            continue
        for r, c in move.cells:
            mf.click(r * cfg.cols + c)
    else:
        pytest.fail(f"seed {seed}: game did not finish")
    return mf


@pytest.mark.parametrize("seed", range(24))
def test_chording_does_not_change_outcomes(seed):
    plain = _play(seed, use_chords=False)
    chorded = _play(seed, use_chords=True)

    assert chorded.won == plain.won
    assert chorded.exploded == plain.exploded
    assert chorded.visible.tolist() == plain.visible.tolist(), (
        "chording revealed a different set of cells - the 'execution-only' claim is false"
    )


def test_min_chord_is_two_cells():
    assert MIN_CHORD_CELLS == 2
