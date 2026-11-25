from argparse import Namespace
from dataclasses import asdict

import numpy as np
import pytest

from ms.board import Board
from ms.engine import Config, EXPERT, Minefield
from ms.eval import play_game
from ms.solver import Move, Solver

torch = pytest.importorskip("torch")
from ms.belief_rl import (BeliefActor, BeliefPolicy, Decision, ReturnActor, batch_inputs,
                          group_rollouts, loo_advantages, observe)
from experiments.belief_rl import optimize


def test_solver_belief_only_on_a_proven_guess():
    solver = Solver(max_nodes=None)
    belief = {(0, 0): 123}
    guess = Board(1, 4, np.full((1, 4), -1, np.int8), 1)
    assert solver.deduce(guess, belief=belief) is None
    assert belief == {(0, c): 0.25 for c in range(4)}
    obs = observe(guess, belief, 4)
    assert np.allclose(obs.planes[4][0], [1, 1 / 3, 1 / 3, 1])
    known = Board(1, 2, np.array([[0, -1]], np.int8), 0)
    assert solver.deduce(known, belief=belief).kind == "open"
    assert belief == {}


def test_raw_control_has_no_conditional_risk_and_actor_respects_reflections():
    torch.manual_seed(9)
    board = Board(2, 4, np.array([[-1, 1, -1, -1], [-1, -2, -3, -1]], np.int8), 2)
    obs = observe(board, None, 7)
    assert np.all(obs.planes[0][board.covered()] == np.float32(2 / 6))
    assert not obs.planes[2:].any()
    net = BeliefActor()
    cells, planes = batch_inputs([obs])
    assert torch.all(net(cells, planes)[0, board.covered().ravel()] == 0)
    with torch.no_grad():
        net.readout.weight.normal_()
        logits = net(cells, planes).reshape(1, 2, 4)
        for dims in ((1,), (2,), (1, 2)):
            reflected = net(cells.flip(dims), planes.flip([d + 1 for d in dims])).reshape(1, 2, 4)
            torch.testing.assert_close(reflected.flip(dims), logits)
        assert np.array_equal(torch.isfinite(logits[0]).numpy(), board.covered())


def test_group_baseline_and_zero_signal_update():
    assert np.array_equal(loo_advantages([1, 1, 1, 1]), np.zeros(4))
    assert np.allclose(loo_advantages([-1, 1, -1, -1]), [-2 / 3, 2, -2 / 3, -2 / 3])
    actor = BeliefActor()
    before = [p.detach().clone() for p in actor.parameters()]
    assert optimize(actor, torch.optim.Adam(actor.parameters()), [], 4, 3) == 0
    assert all(torch.equal(a, b) for a, b in zip(before, actor.parameters()))


@pytest.mark.parametrize("won", [False, True])
def test_terminal_value_respects_survival_and_learns_only_the_observed_result(won):
    position = Board(2, 4, np.full((2, 4), -1, np.int8), 2)
    belief = {rc: 0.25 for rc in np.ndindex(position.cells.shape)}
    obs = observe(position, belief, 28)
    net = ReturnActor()
    def value():
        with torch.no_grad():
            return float(net(*batch_inputs([obs]))[0, 0].exp())
    assert value() == pytest.approx(.375)
    optimize(net, torch.optim.Adam(net.parameters(), lr=.001), [(Decision(obs, 0, 0), int(won))], 1, 1)
    assert (value() > .375) == won
    assert 0 < value() <= .75


def test_large_state_digit_predictions_use_only_visible_information():
    position = Board(6, 8, np.full((6, 8), -1, np.int8), 10)
    belief = {rc: 10 / 48 for rc in np.ndindex(position.cells.shape)}
    first = observe(position, belief, 10**8, sample_large=True)
    second = observe(position, belief, 10**8, sample_large=True)
    np.testing.assert_array_equal(first.planes, second.planes)
    assert (first.planes[2] == -1).all()
    assert (first.planes[3] >= 0).all()
    assert ((first.planes[4:] >= 0) & (first.planes[4:] <= 1)).all()


def test_terminal_rollout_actor_improves_on_its_own_continuation():
    from ms.belief_search import RolloutActor
    from tests.test_endgame import RISK_TRADEOFF

    torch.set_num_threads(1)
    actor = BeliefActor()  # uniform initial actor breaks ties in row-major order
    assert BeliefPolicy(actor, 3).act(RISK_TRADEOFF).cells == ((0, 0),)
    policy = RolloutActor(actor, 3)
    assert policy.act(RISK_TRADEOFF).cells == ((2, 2),)
    assert (policy.searches, policy.switches) == (1, 1)


def test_grouped_rollouts_reproduce_single_click_evaluator():
    torch.set_num_threads(1)
    cfg = Config(6, 6, 8)
    actor = BeliefActor()
    count = 0
    for seed in range(10):
        grouped = group_rollouts(cfg, seed, actor, 2, 42)
        for i, (reward, decisions) in enumerate(grouped):
            rng = torch.Generator().manual_seed(
                int(np.random.SeedSequence([42, seed, i]).generate_state(1, dtype=np.uint64)[0]))
            actions = []

            class Sampled(BeliefPolicy):
                def act(self, board):
                    state = self.position(board)
                    if isinstance(state, Move):
                        return state
                    chosen = self.choose(state, rng)
                    actions.append(chosen.cell)
                    return Move("guess", (divmod(chosen.cell, board.cols),), None, "test")

            result = play_game(seed, Sampled(actor, seed), cfg, opening="random")
            assert result.won == (reward == 1)
            assert actions == [d.cell for d in decisions]
            count += len(actions)
    assert count > 0


def test_all_evaluation_arms_play_the_same_layout(tmp_path, monkeypatch):
    from experiments.belief_rl import compare
    from ms.rl import BoardPolicyNet

    torch.set_num_threads(1)
    paths = {}
    common = dict(cfg=asdict(EXPERT), seed_base=100, opening="random")
    for name, net, metadata in [
        ("checkpoint", BeliefActor(), dict(layouts=10)),
        ("old_checkpoint", BoardPolicyNet(), dict(games=10)),
        ("raw_checkpoint", BeliefActor(), dict(layouts=10, raw_only=True)),
        ("value_checkpoint", ReturnActor(), dict(layouts=10, value=True)),
    ]:
        paths[name] = tmp_path / (name + ".pt")
        torch.save(dict(model=net.state_dict(), training=dict(common, **metadata)), paths[name])
    args = Namespace(**paths, jobs=1, games=1, seed_base=109, rollout=True)
    with pytest.raises(ValueError, match="overlap"):
        compare(args)
    args.seed_base = 110
    placed = []
    place = Minefield._place

    def record_layout(field, first):
        place(field, first)
        placed.append((first, field.mines.tobytes()))

    monkeypatch.setattr(Minefield, "_place", record_layout)
    result = compare(args)
    assert len(placed) == len(result["policies"]) == 7
    assert len(set(placed)) == 1
    assert result["boards"][0]["first_cell"] == placed[0][0]
    assert all(p["pairs"] == 1 for p in result["paired"])
