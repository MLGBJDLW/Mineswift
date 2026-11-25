"""Warm-start one terminal-reward actor; compare checkpoints on untouched paired seeds."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiments.belief_rl import collect, cpu_weights, init_worker, optimize, pool_for, restore
from ms.belief_rl import BeliefActor
from ms.belief_search import RolloutActor
from ms.engine import EXPERT
from ms.eval import mcnemar, play_game, summarize


def evaluate(task):
    seed, initial, updated = task
    return [play_game(seed, RolloutActor(restore(BeliefActor(), weights), seed), EXPERT, opening="random")
            for weights in (initial, updated)]


def train_batch(dispatch, actor, optimizer, seeds, group):
    weights = cpu_weights(actor)
    tasks = [(weights, seed, group, 30, False, False) for seed in seeds]
    data = list(dispatch(collect, tasks))
    samples = [item for _,_,actions in data for item in actions]
    epochs = []
    optimize(actor, optimizer, samples, len(seeds)*group, 3, epochs)
    return dict(layouts=len(seeds), wins=sum(w for w,_,_ in data), samples=len(samples), epochs=epochs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layouts", type=int, default=256)
    p.add_argument("--games", type=int, default=512)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    init_worker()
    src = Path("models/belief_rl.pt")
    base = torch.load(src, weights_only=True, map_location="cpu")
    seed_train, seed_test, group, batch = 670000, 680000, 4, 16
    if min(a.jobs, a.games, a.layouts) < 1 or seed_train + a.layouts > seed_test:
        p.error("positive counts and disjoint seed ranges required")
    actor = BeliefActor()
    actor.load_state_dict(base["model"])
    initial = cpu_weights(actor)
    lr = .002
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    log = []
    with pool_for(a.jobs) as pool:
        dispatch = pool.map if pool else map
        for start in range(0, a.layouts, batch):
            n = min(a.layouts-start, batch)
            log.append(train_batch(dispatch, actor, optimizer, range(seed_train+start, seed_train+start+n), group))
        updated = cpu_weights(actor)
        results = list(dispatch(evaluate, [(seed_test+i, initial, updated) for i in range(a.games)]))
    stats = [summarize(name, EXPERT, games, 0) for name,games in zip(("live", "continued"), zip(*results))]
    report = dict(initial_sha256=hashlib.sha256(src.read_bytes()).hexdigest(), lr=lr, fresh_adam=True,
                  seed_base=seed_train, layouts=a.layouts, episodes=group*a.layouts,
                  eval_seed_base=seed_test, games=a.games, opening="random", batches=log,
                  wins=[s.wins for s in stats], paired=mcnemar(*stats),
                  boards=[dict(seed=s[0].seed, live=s[0].outcome, continued=s[1].outcome) for s in results])
    a.out.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("batches","boards")}))


if __name__ == "__main__":
    main()
