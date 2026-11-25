"""Check the solver/model boundary, terminal credit and the paired experiment."""
import numpy as np
import pytest

from ms.board import Board, COVERED as C, FLAG as F, QUESTION as Q
from ms.engine import Config, Minefield
from ms.opening import first_cell
from ms.solver import BudgetExceeded

torch = pytest.importorskip("torch")
from ms.rl import BoardPolicyNet, RLGuessPolicy, compare, episode_loss, train


def board(cells, mines_left):
    cells = np.array(cells, dtype=np.int8)
    return Board(*cells.shape, cells, mines_left)


@pytest.mark.parametrize(("position", "kind", "cells"), [
    (board([[0, C]], 0), "open", ((0, 1),)),
    (board([[1, Q]], 1), "flag", ((0, 1),)),
    (board([[C, 1, C, C]], 1), "open", ((0, 3),)),  # global mine budget
    (board([[C, 1, C, C]], 2), "flag", ((0, 3),)),
    (board([[0]], 0), "done", ()),
])
def test_solver_decisions_bypass_model(position, kind, cells):
    def unexpected(*_):
        pytest.fail("model called on a deduction")

    policy = RLGuessPolicy(0, unexpected)
    move = policy.act(position)
    assert (move.kind, move.cells) == (kind, cells)
    assert policy.guesses == 0


def test_guess_can_select_outside_frontier_and_above_minimum_risk():
    position = board([[C, 1, C, C, C, Q]], 2)
    for index in (0, 5):
        def pick(visible):
            assert visible is position
            scores = torch.zeros(visible.cells.size)
            scores[index] = 9
            return scores

        policy = RLGuessPolicy(0, pick)
        assert policy.act(position).cells == ((0, index),)
        assert policy.log_probs == []


def test_budget_exhaustion_never_authorizes_a_guess(monkeypatch):
    position = board([[C, 1, C, C, C]], 2)
    policy = RLGuessPolicy(0, BoardPolicyNet())
    policy.solver.max_nodes = 0
    monkeypatch.setattr("ms.solver.sample_group", lambda *_: pytest.fail("sampling a proof"))
    with pytest.raises(BudgetExceeded):
        policy.act(position)
    assert policy.guesses == 0


def test_full_board_logits_mask_and_remote_context():
    torch.manual_seed(7)
    model = BoardPolicyNet()
    position = board([[C, Q, 2, F, 0, 1, C, C, C, C, C]], 2)
    with torch.no_grad():
        logits = model(position)
        assert logits.shape == (position.cols,)
        np.testing.assert_array_equal(torch.isfinite(logits).numpy(), position.covered().ravel())
        # Outside both 3x3 convolutions' local field, still affects relative logits.
        position.cells[0, -1] = 3
        altered = model(position)
        assert (logits[0] - logits[1]).item() != (altered[0] - altered[1]).item()


@pytest.mark.parametrize("won", [False, True])
def test_only_terminal_outcome_controls_credit_for_all_guesses(won):
    logits = torch.zeros((2, 3), requires_grad=True)
    log_probs = logits.log_softmax(dim=1)
    episode_loss([log_probs[0, 1], log_probs[1, 2]], won).backward()
    for t, action in enumerate([1, 2]):
        assert (logits.grad[t, action] < 0).item() == won
        assert (logits.grad[t, 0] > 0).item() == won


def test_training_checkpoint_and_independent_random_pairs(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    cfg = Config(6, 6, 8)
    model, training = train(cfg, games=16, seed_base=100, model_seed=42, batch=4, lr=0.003)
    assert training["updates"] and training["parameter_delta_l2"] > 0
    assert training["guesses"] >= training["games_with_guesses"] > 0
    path = tmp_path / "rl.pt"
    torch.save({"model": model.state_dict(), "training": training}, path)
    with pytest.raises(ValueError, match="overlap"):
        compare(path, games=3, seed_base=98)
    layouts = []
    place = Minefield._place

    def record_layout(game, first):
        place(game, first)
        layouts.append((first, game.mines.tobytes()))

    monkeypatch.setattr(Minefield, "_place", record_layout)
    report = compare(path, games=5, seed_base=116)
    assert report["paired"]["pairs"] == len(report["boards"]) == 5
    assert len(layouts) == 10 and layouts[:5] == layouts[5:]
    assert [first for first, _ in layouts[:5]] == [row["first_cell"] for row in report["boards"]]
    assert [(row["seed"], row["first_cell"]) for row in report["boards"]] == [
        (s, first_cell(cfg, s, "random")) for s in range(116, 121)]
    assert all(row[name] in ("win", "guess") for row in report["boards"] for name in ("endgame", "rl"))
