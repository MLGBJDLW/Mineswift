"""Regression tests for the exact core. These are the tests that matter: every one of them
compares the solver against brute force, so a refactor that breaks exactness fails here
instead of quietly costing win rate in the evaluator.
"""
from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

from ms.board import COVERED, FLAG, QUESTION, Board
from ms.engine import Config, Minefield, neighbour_ids
from ms.solver import Inconsistent, Solver, Constraint, count_group_dp, enumerate_group, raw_constraints, simplify


def brute(cells, group):
    """All assignments of `cells` satisfying `group` - the definition of truth here."""
    out = []
    for bits in itertools.product([0, 1], repeat=len(cells)):
        st = dict(zip(cells, bits))
        if all(sum(st[x] for x in c.cells) == c.mines for c in group):
            out.append(st)
    return out


def random_group(rng, n_cells, n_cons):
    cells = list(range(100, 100 + n_cells))
    group = []
    for _ in range(n_cons):
        k = rng.randint(1, n_cells)
        group.append(Constraint(tuple(sorted(rng.sample(cells, k))), rng.randint(0, k)))
    return cells, group


@pytest.mark.parametrize("trial", range(120))
def test_enumeration_matches_brute_force(trial):
    """The enumerated counts must equal the counts a brute-force solver computes.

    Note the contract: a cell that no constraint mentions is not part of the component's
    count (it is another component's business, or free), so the brute-force reference is
    taken over the mentioned cells only.
    """
    rng = random.Random(trial)
    cells, group = random_group(rng, rng.randint(1, 8), rng.randint(1, 3))
    # brute force only over the cells the constraints mention: the others are free and are
    # deliberately not part of this component's count
    mentioned = sorted({x for c in group for x in c.cells})
    ref = brute(mentioned, group)
    if not ref:
        with pytest.raises(Inconsistent):
            count_group_dp(group, 4)
        pytest.skip("unsatisfiable constraint set")
    gc, succ_by_k, hist, n_sol, _ = enumerate_group(group, 10 ** 7)
    assert count_group_dp(group, 4)[:4] == (gc, succ_by_k, hist, n_sol)
    expected_hist: dict[int, int] = {}
    for st in ref:
        m = sum(v for k, v in st.items() if k in mentioned)
        expected_hist[m] = expected_hist.get(m, 0) + 1
    assert hist == expected_hist
    assert n_sol == sum(expected_hist.values())
    assert set(gc) == set(mentioned)
    for j, cell in enumerate(gc):
        assert sum(succ_by_k[j].values()) == sum(1 for st in ref if st[cell] == 1)
        # and the per-k split is exact too, not just the total
        for m, count in succ_by_k[j].items():
            want = sum(1 for st in ref if st[cell] == 1
                       and sum(v for k, v in st.items() if k in mentioned) == m)
            assert count == want


def test_large_exact_sweep_counts_marginals_without_listing_worlds():
    # For each fixed value of the hub, the other pairs admit 2**35 or 1 completions.
    group = [Constraint((0, 2*i + 1, 2*i + 2), 1) for i in range(35)]
    _, mine, hist, _, _ = count_group_dp(group, 72)
    assert hist == {35: 2**35, 1: 1}
    assert mine[0] == {1: 1}
    assert all(m == {35: 2**34} for m in mine[1:])


@pytest.mark.parametrize("trial", range(80))
def test_simplify_preserves_solution_set(trial):
    rng = random.Random(1000 + trial)
    cells, group = random_group(rng, rng.randint(2, 7), rng.randint(1, 3))
    ref = brute(cells, group)
    if not ref:
        pytest.skip("unsatisfiable constraint set")
    simp = simplify(group)
    left = {x for c in simp for x in c.cells}
    before = {frozenset(k for k, v in st.items() if v == 1 and k in left) for st in ref}
    after = ({frozenset(k for k, v in st.items() if v == 1)
              for st in brute(sorted(left), simp)} if simp else {frozenset()})
    assert before == after


def exact_probabilities(visible, cfg, nb, mines_total):
    """Brute force over the whole covered set: the definition of a correct probability."""
    n = cfg.rows * cfg.cols
    cov = [i for i in range(n) if visible[i] == COVERED]
    need = mines_total - int((visible == FLAG).sum())
    cons = []
    for i in range(n):
        v = int(visible[i])
        if v < 0:
            continue
        ns = [j for j in nb[i] if j >= 0]
        cons.append(([j for j in ns if visible[j] == COVERED],
                     v - sum(1 for j in ns if visible[j] == FLAG)))
    succ = {c: 0 for c in cov}
    total = 0
    for bits in itertools.product([0, 1], repeat=len(cov)):
        if sum(bits) != need:
            continue
        st = dict(zip(cov, bits))
        if all(sum(st[x] for x in cl) == k for cl, k in cons):
            total += 1
            for c, b in st.items():
                succ[c] += b
    return None if total == 0 else {c: succ[c] / total for c in cov}


@pytest.mark.parametrize("size,mines", [(5, 6), (6, 8)])
def test_probabilities_match_brute_force(size, mines):
    cfg = Config(size, size, mines)
    nb = neighbour_ids(size, size)
    checked = 0
    for seed in range(120):
        rng = np.random.default_rng(seed)
        mf = Minefield(cfg, rng)
        for _ in range(4):
            mf.click(int(rng.integers(0, size * size)))
            if mf.exploded:
                break
        if mf.exploded or not mf.placed:
            continue
        vis = mf.visible.copy()
        if int((vis == COVERED).sum()) > 15:
            continue
        ref = exact_probabilities(vis, cfg, nb, mines)
        if ref is None:
            continue
        board = Board(size, size, vis.reshape(size, size).copy(), mf.mines_left_counter())
        probs, _ = Solver(seed=seed).measure(board, {})
        model = {r * size + c: probs[(r, c)]
                 for r in range(size) for c in range(size) if vis[r * size + c] == COVERED}
        for cell in ref:
            assert abs(model[cell] - ref[cell]) < 1e-9, (seed, cell)
        checked += 1
    assert checked > 3, "not enough boards exercised"


def test_question_marks_are_openable():
    cells = np.full((5, 5), COVERED, dtype=np.int8)
    cells[2, 2] = 1
    cells[2, 3] = QUESTION
    board = Board(5, 5, cells, 1)
    move = Solver().decide(board)
    covered_or_question = {(r, c) for r in range(5) for c in range(5)
                          if cells[r, c] in (COVERED, QUESTION)}
    assert all(cell in covered_or_question for cell in move.cells)


def test_impossible_counter_is_reported_not_guessed():
    """A board that cannot exist on screen must raise, never produce a move."""
    cells = np.full((5, 5), COVERED, dtype=np.int8)
    cells[2, 2] = 1
    cells[2, 3] = FLAG
    cells[2, 1] = FLAG
    board = Board(5, 5, cells, 0)
    with pytest.raises(Inconsistent):
        Solver().decide(board)


def test_decide_is_deterministic():
    cfg = Config(9, 9, 10)
    mf = Minefield(cfg, np.random.default_rng(7))
    mf.click(40)
    board = mf.board()
    a = Solver(seed=0).decide(board)
    b = Solver(seed=0).decide(board)
    assert a == b
