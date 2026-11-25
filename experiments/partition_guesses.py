"""Compare a larger exact endgame search with the frozen, previously verified policy."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from experiments.belief_rl import init_worker
from experiments.midgame_returns import frozen_actor
from experiments.swap_guesses import SwapActor
from ms.belief_rl import batch_inputs
from ms.belief_search import RolloutActor
from ms.endgame import _worlds
from ms.engine import EXPERT
from ms.eval import GameResult, mcnemar, play_game, summarize
from ms.play_web import WEIGHTS
from ms.policy import EndgamePolicy
from ms.solver import BudgetExceeded, Move
from ms.verified_search import VerifiedActor
from ms.world_partitions import WorldPartitions


class PartitionActor(VerifiedActor):
    name = "belief_partitions"

    def __init__(self, actor, seed):
        super().__init__(actor, seed)
        self.expanded = self.rechecked = self.review_switches = 0

    @torch.no_grad()
    def act(self, board):
        complete = self.complete
        incumbent = super().act(board)
        if incumbent.kind != "guess" or self.complete > complete:
            return incumbent
        count = self.solver.last_stats["solutions"]
        if not 2 < count <= 2048:
            return incumbent
        hidden = tuple(map(int, np.flatnonzero(board.covered())))
        if len(hidden) > 52:
            return incumbent
        self.expanded += 1
        try:
            worlds = _worlds(board, hidden, count, 65_536)
            tree = WorldPartitions(board, hidden, worlds, 250_000)
            first = hidden.index(int(np.ravel_multi_index(incumbent.cells[0], board.cells.shape)))
            scores = self.actor(*batch_inputs([self.current]))[0].numpy()
            order = sorted(range(len(hidden)), key=lambda i: scores[hidden[i]], reverse=True)
            result = tree.compare(first, order)
            if result is None:
                return incumbent
            self.rechecked += 1
            candidate, _, wins, _ = result
            self.last_guess.update(verified_worlds=count, optimal_wins=wins)
            if candidate == first:
                return incumbent
            self.review_switches += 1
            cell = hidden[candidate]
            risk = float(self.current.planes[0].flat[cell])
            self.last_guess.update(mine_risk=risk, rollout_worlds=0, rollout_wins=None)
            return Move("guess", (divmod(cell, board.cols),), risk, "model candidate with higher exact terminal value")
        except BudgetExceeded:
            return incumbent


def evaluate(seed, policy_class=PartitionActor):
    actor = frozen_actor()
    def run(policy):
        result = play_game(seed, policy, EXPERT, opening="random")
        if result.outcome not in ("win", "guess"):
            raise RuntimeError(f"unexpected outcome {result.outcome} on {seed}")
        return result.won

    advanced = policy_class(actor, seed)
    result = run(advanced)
    verified = VerifiedActor(actor, seed) if advanced.review_switches else advanced
    previous = run(verified) if advanced.review_switches else result
    live = run(RolloutActor(actor, seed)) if verified.changes else previous
    return dict(seed=seed, advanced=result, verified=previous, live=live, endgame=run(EndgamePolicy(seed)),
                attempts=advanced.expanded, completed=advanced.rechecked, switches=advanced.review_switches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--method", choices=("partition", "swap"), default="partition")
    args = parser.parse_args()
    if min(args.games, args.jobs) < 1 or args.seed < 0:
        parser.error("positive counts and nonnegative seeds required")
    with ProcessPoolExecutor(args.jobs, initializer=init_worker) as pool:
        rows = []
        work = partial(evaluate, policy_class=PartitionActor if args.method == "partition" else SwapActor)
        for row in pool.map(work, range(args.seed, args.seed+args.games), chunksize=4):
            rows.append(row)
            if len(rows) % 256 == 0:
                print(len(rows), flush=True)
    stats = {}
    for name in ("advanced", "verified", "live", "endgame"):
        records = [GameResult(r["seed"], r[name], 0, "win" if r[name] else "guess") for r in rows]
        stats[name] = summarize(name, EXPERT, records, 0)
    report = dict(seed=args.seed, games=args.games, cfg=asdict(EXPERT), opening="random", method=args.method,
                  actor_sha256=sha256(WEIGHTS.read_bytes()).hexdigest(),
                  wins={k:v.wins for k,v in stats.items()}, changed=sum(bool(r["switches"]) for r in rows),
                  verified_vs_live=mcnemar(stats["live"], stats["verified"]),
                  paired=[mcnemar(stats[k], stats["advanced"]) for k in ("verified", "live", "endgame")])
    print(report, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(**report, rows=rows), separators=(",", ":"))+"\n")


if __name__ == "__main__":
    main()
