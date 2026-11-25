"""Paired expert replay and post-mortem of the actor used by the browser player."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.belief_rl import init_worker, pool_for, restore
from experiments.loss_audit import swap_witness
from ms.belief_rl import BeliefActor
from ms.belief_search import RolloutActor
from ms.endgame import _Search, _observations, _worlds
from ms.engine import EXPERT, Minefield
from ms.eval import GameResult, mcnemar, play_game, summarize
from ms.opening import first_cell
from ms.policy import EndgamePolicy
from ms.solver import BudgetExceeded


def exact_gap(board, cell, count):
    hidden = tuple(map(int, np.flatnonzero(board.covered())))
    if not 1 < count <= 96 or len(hidden) > 32:
        return None
    try:
        worlds = _worlds(board, hidden, count, 6000)
        search = _Search(worlds, _observations(board, hidden, worlds), board.mines_left, len(hidden), 150000)
        remaining, feasible = (1 << len(hidden))-1, (1 << count)-1
        i = hidden.index(cell)
        chosen = search._click(remaining, feasible & ~search.mine_worlds[i], i)
        _, best = search.pick(remaining, feasible, i)
        return (best - chosen) / count
    except BudgetExceeded:
        return None


def replay(task):
    seed, weights = task
    net = restore(BeliefActor(), weights)
    first = first_cell(EXPERT, seed, "random")
    row = dict(seed=seed, first_cell=first)
    for name, options in (("live", {}), ("wider", dict(world_limit=64, cell_limit=32, candidate_limit=12))):
        field = Minefield(EXPERT, np.random.default_rng(seed * 1_000_003 + 7))
        field.click(first)
        policy = RolloutActor(net, seed, **options)
        start = perf_counter()
        steps = 0
        while not field.won and not field.exploded:
            board = field.board()
            move = policy.act(board)
            steps += 1
            if steps > 4000 or not move.cells:
                raise RuntimeError(f"unfinished board {seed}")
            if move.kind == "flag":
                for r, c in move.cells:
                    field.flag(r * board.cols + c)
                continue
            r, c = move.cells[0]
            cell = r * board.cols + c
            if field.click(cell) == "boom":
                if move.kind != "guess":
                    raise RuntimeError(f"loss on a proven move: {seed}")
                if name == "live":
                    row["death"] = dict(cell=cell, **policy.last_guess,
                                        indistinguishable_partner=swap_witness(field, cell, first),
                                        optimal_win_gap=exact_gap(board, cell, policy.solver.last_stats["solutions"]))
        row[name] = "win" if field.won else "guess"
        row[name + "_seconds"] = round(perf_counter() - start, 4)
    row["endgame"] = play_game(seed, EndgamePolicy(seed), EXPERT, opening="random").outcome
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--seed-base", type=int, default=640000)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    init_worker()
    path = Path("models/belief_rl.pt")
    saved = torch.load(path, weights_only=True, map_location="cpu")
    training = saved["training"]
    if args.games < 1 or args.jobs < 1 or args.seed_base < 0 or max(args.seed_base, training["seed_base"]) < min(
            args.seed_base + args.games, training["seed_base"] + training["layouts"]):
        parser.error("provide positive counts and independent board seeds")
    weights = {k: v.numpy() for k, v in saved["model"].items()}
    tasks = [(args.seed_base + i, weights) for i in range(args.games)]
    with pool_for(args.jobs) as pool:
        rows = list(pool.map(replay, tasks) if pool else map(replay, tasks))
    stats = {}
    for name in ("live", "wider", "endgame"):
        results = [GameResult(r["seed"], r[name] == "win", 0, r[name]) for r in rows]
        stats[name] = summarize(name, EXPERT, results, 0)
    losses = [r["death"] for r in rows if "death" in r]
    gaps = [d["optimal_win_gap"] for d in losses if d["optimal_win_gap"] is not None]
    report = dict(games=args.games, seed_base=args.seed_base, cfg=asdict(EXPERT), opening="random",
                  weights_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  outcomes={n: dict(Counter(r[n] for r in rows)) for n in stats},
                  paired=[mcnemar(stats["live"], stats[n]) for n in ("wider", "endgame")],
                  live_losses=dict(total=len(losses), swap_witness=sum(d["indistinguishable_partner"] is not None for d in losses),
                                   higher_immediate_risk=sum(d["mine_risk"] > d["min_risk"]+1e-6 for d in losses),
                                   exactly_solved=len(gaps), suboptimal_terminal_action=sum(g > 0 for g in gaps)),
                  boards=rows)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "boards"}))


if __name__ == "__main__":
    main()
