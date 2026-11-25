"""Guard rails for the learned scorer: it must never be able to play worse than the solver.

These tests are cheap and they encode the promise the README makes, so a future refactor that
lets the model override a logical move fails here.
"""
from __future__ import annotations

import numpy as np
import pytest

from ms.board import COVERED, Board
from ms.engine import Config, Minefield
from ms.policy import ExactSolverPolicy
from ms.solver import Solver, certain_moves, raw_constraints


def test_model_policy_never_overrides_a_certain_move():
    """If a zero-risk move exists, the model must not be consulted at all."""
    torch = pytest.importorskip("torch")  # noqa: F841  (skip when torch is absent)
    from ms.policy import LearnedGuessPolicy  # deferred: needs torch

    cfg = Config(9, 9, 10)
    mf = Minefield(cfg, np.random.default_rng(1))
    mf.click(40)
    policy = LearnedGuessPolicy(seed=0, model=None, weights=None)
    for _ in range(20):
        board = mf.board()
        raw, _, _ = raw_constraints(board)
        safe, mines = certain_moves(raw)
        move = policy.act(board)
        if safe:
            assert move.kind == "open"
            assert set(move.cells) <= {(x // cfg.cols, x % cfg.cols) for x in safe}
        else:
            assert move.kind in ("open", "flag", "guess", "done")
        if move.kind in ("open", "guess"):
            r, c = move.cells[0]
            if mf.click(r * cfg.cols + c) == "boom":
                break
        elif move.kind == "flag":
            for r, c in move.cells:
                mf.flag(r * cfg.cols + c)


def test_learned_policy_without_weights_falls_back_to_exact():
    """A missing/broken model must degrade to the exact solver, never to a random choice."""
    pytest.importorskip("torch")
    from ms.policy import LearnedGuessPolicy

    cfg = Config(9, 9, 10)
    mf = Minefield(cfg, np.random.default_rng(3))
    mf.click(40)
    board = mf.board()
    learned = LearnedGuessPolicy(seed=0, model=None, weights=None).act(board)
    exact = ExactSolverPolicy(seed=0).act(board)
    assert learned == exact
