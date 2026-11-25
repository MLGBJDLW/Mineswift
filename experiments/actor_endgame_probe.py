"""Compare exact observation-tree scores for actor-proposed guesses on new expert boards."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path

from ms.belief_search import RolloutActor
from ms.endgame import _observations, _Search
from ms.engine import EXPERT
from ms.eval import GameResult, mcnemar, play_game, summarize
from ms.opening import first_cell
from ms.play_web import WEIGHTS, load_policy
from ms.solver import BudgetExceeded


class TreeActor(RolloutActor):
    def _terminal_wins(self, board, hidden, worlds, candidates):
        search = _Search(
            worlds,
            _observations(board, hidden, worlds),
            board.mines_left,
            len(hidden),
            100_000,
        )
        remaining, feasible = (1 << len(hidden)) - 1, (1 << len(worlds)) - 1
        try:
            scores = []
            for cell in candidates:
                i = hidden.index(cell)
                scores.append(
                    search._click(remaining, feasible & ~search.mine_worlds[i], i)
                )
            return scores
        except BudgetExceeded:
            return super()._terminal_wins(board, hidden, worlds, candidates)


def pair(seed):
    live = load_policy(seed)
    tree = TreeActor(live.actor, seed, world_limit=48, cell_limit=28)
    return dict(
        seed=seed,
        first=first_cell(EXPERT, seed, "random"),
        live=play_game(seed, live, EXPERT).outcome,
        tree=play_game(seed, tree, EXPERT).outcome,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-base", type=int, default=925000)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with ProcessPoolExecutor(args.jobs) as pool:
        boards = list(
            pool.map(
                pair, range(args.seed_base, args.seed_base + args.games), chunksize=4
            )
        )
    stats = [
        summarize(
            name,
            EXPERT,
            [GameResult(b["seed"], b[name] == "win", 0, b[name]) for b in boards],
            0,
        )
        for name in ("live", "tree")
    ]
    result = dict(
        shape="16x30/99",
        opening="random",
        seed_base=args.seed_base,
        games=len(boards),
        weights_sha256=sha256(WEIGHTS.read_bytes()).hexdigest(),
        wins={s.policy: s.wins for s in stats},
        paired=mcnemar(*stats),
    )
    args.out.write_text(
        json.dumps(dict(result, boards=boards), indent=2) + "\n", encoding="utf8"
    )
    print(json.dumps(result))
