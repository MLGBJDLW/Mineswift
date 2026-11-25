"""Replay the frozen web policy against endgame on identical expert boards."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch

from ms.engine import EXPERT
from ms.eval import GameResult, mcnemar, play_game, summarize
from ms.opening import first_cell
from ms.play_web import WEIGHTS, load_policy
from ms.policy import EndgamePolicy


def pair(seed):
    live = play_game(seed, load_policy(seed, review=False), EXPERT, opening="random")
    endgame = play_game(seed, EndgamePolicy(seed=seed), EXPERT, opening="random")
    return dict(
        seed=seed,
        first_cell=first_cell(EXPERT, seed, "random"),
        live=live.outcome,
        endgame=endgame.outcome,
    )


def summary(rows):
    stats = []
    for name in ("endgame", "live"):
        results = [GameResult(r["seed"], r[name] == "win", 0, r[name]) for r in rows]
        stats.append(summarize(name, EXPERT, results, 0))
    return {
        s.policy: {
            "wins": s.wins,
            "games": s.games,
            "ci95": s.ci95,
            "logic_losses": s.losses_logic,
            "stuck": s.stuck,
        }
        for s in stats
    } | {"paired": mcnemar(*stats)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-base", type=int, default=830000)
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    saved = torch.load(WEIGHTS, map_location="cpu", weights_only=True)["training"]
    start, end = args.seed_base, args.seed_base + args.games
    train = saved["seed_base"]
    if (
        min(args.jobs, args.games) < 1
        or start < 0
        or max(start, train) < min(end, train + saved["layouts"])
    ):
        parser.error("require positive counts and seeds outside training")
    seeds = range(start, end)
    if args.jobs == 1:
        rows = list(map(pair, seeds))
    else:
        with ProcessPoolExecutor(args.jobs) as pool:
            rows = list(pool.map(pair, seeds, chunksize=4))
    report = dict(
        cfg=asdict(EXPERT),
        opening="random",
        seed_base=start,
        games=args.games,
        weights_sha256=sha256(WEIGHTS.read_bytes()).hexdigest(),
        policy="belief_rollout",
        runtime=dict(
            python=sys.version.split()[0],
            numpy=np.__version__,
            torch=str(torch.__version__),
            platform=platform.system(),
            jobs=args.jobs,
        ),
        **summary(rows),
        boards=rows,
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "boards"}))


if __name__ == "__main__":
    main()
