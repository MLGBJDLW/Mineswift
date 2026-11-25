"""Offline audit of the first forced guess, before any learned action influences the board."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.belief_rl import init_worker, pool_for, restore
from ms.belief_rl import BeliefActor, BeliefPolicy, ReturnActor, next_guess
from ms.deadguess import forced_pair
from ms.endgame import _Search, _observations, _worlds
from ms.engine import EXPERT, Minefield
from ms.opening import first_cell
from ms.solver import BudgetExceeded


def audit(task):
    seed, weights, raw_only, value = task
    field = Minefield(EXPERT, np.random.default_rng(seed * 1_000_003 + 7))
    field.click(first_cell(EXPERT, seed, "random"))
    actor = restore(ReturnActor() if value else BeliefActor(), weights)
    policy = BeliefPolicy(actor, seed, raw_only=raw_only)
    obs = next_guess(field, policy)
    if obs is None:
        return dict(seed=seed, kind="solver_win", upper=1.)
    board = field.board()
    count = policy.solver.last_stats["solutions"]
    hidden = tuple(map(int, np.flatnonzero(board.covered())))
    cell = policy.choose(obs)
    # Reuse exact double-precision probabilities for the mathematical upper bound.
    risk, _ = policy.solver.measure(board, exact_only=True)
    lowest = min(risk.values())
    row = dict(seed=seed, min_risk=lowest, chosen_risk=risk[divmod(cell, board.cols)],
               kind="first_click_survival", upper=1 - lowest)
    if forced_pair(board):
        row.update(kind="unavoidable_pair", upper=min(0.5, row["upper"]))
    if 1 < count <= 160 and len(hidden) <= 48:
        try:
            worlds = _worlds(board, hidden, count, 6000)
            search = _Search(worlds, _observations(board, hidden, worlds),
                             board.mines_left, len(hidden), 200000)
            remaining = (1 << len(hidden)) - 1
            feasible = (1 << count) - 1
            i = hidden.index(cell)
            won = search._click(remaining, feasible & ~search.mine_worlds[i], i)
            _, best = search.pick(remaining, feasible, i)
            row.update(kind="exact_endgame", upper=best / count, chosen_then_optimal=won / count)
        except BudgetExceeded:
            pass
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=512)
    ap.add_argument("--seed-base", type=int, default=520100)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--checkpoint", type=Path, default=Path("models/belief_rl.pt"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    init_worker()
    checkpoint = torch.load(args.checkpoint, weights_only=True)
    training = checkpoint["training"]
    if args.games < 1 or max(args.seed_base, training["seed_base"]) < min(
            args.seed_base + args.games, training["seed_base"] + training["layouts"]):
        raise ValueError("positive audit size and independent board seeds required")
    weights = {k: v.numpy() for k, v in checkpoint["model"].items()}
    tasks = [(s, weights, training.get("raw_only", False), training.get("value", False))
             for s in range(args.seed_base, args.seed_base + args.games)]
    with pool_for(args.jobs) as pool:
        rows = list(pool.map(audit, tasks) if pool else map(audit, tasks))
    upper = np.asarray([r["upper"] for r in rows])
    exact = [r for r in rows if r["kind"] == "exact_endgame"]
    large = [r for r in rows if r["kind"] in ("first_click_survival", "unavoidable_pair")]
    report = dict(seed_base=args.seed_base, games=args.games, opening="random",
                  solver_wins=sum(r["kind"] == "solver_win" for r in rows),
                  exact_endgames=len(exact), other_guesses=len(large),
                  mean_upper=float(upper.mean()),
                  upper_mean_se=float(upper.std(ddof=1) / np.sqrt(len(upper))) if len(upper) > 1 else None,
                  exact_mean_first_action_win_gap=float(np.mean([
                      r["upper"] - r["chosen_then_optimal"] for r in exact])) if exact else None,
                  other_mean_first_click_risk_above_min=float(np.mean([
                      r["chosen_risk"] - r["min_risk"] for r in large])) if large else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(report, boards=rows), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
