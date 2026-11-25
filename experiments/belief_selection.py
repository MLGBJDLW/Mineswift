"""One predeclared checkpoint selection with a one-use, independent final test."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiments.belief_rl import cpu_weights, init_worker, pool_for, restore
from experiments.continuation_probe import train_batch
from ms.belief_rl import BeliefActor
from ms.belief_search import RolloutActor
from ms.engine import EXPERT
from ms.eval import mcnemar, play_game, summarize
from ms.policy import EndgamePolicy


def compare_one(task):
    seed, weights, endgame = task
    policies = [RolloutActor(restore(BeliefActor(), w), seed) for w in weights]
    if endgame:
        policies.append(EndgamePolicy(seed))
    return [play_game(seed, p, EXPERT, opening="random") for p in policies]


def compare(dispatch, seeds, names, weights, final=False):
    rows = list(dispatch(compare_one, [(s, weights, final) for s in seeds]))
    stats = [summarize(n, EXPERT, r, 0) for n,r in zip(names, zip(*rows))]
    return dict(wins={s.policy:s.wins for s in stats}, paired=[mcnemar(stats[0], s) for s in stats[1:]],
                invalid={s.policy:s.stuck+s.losses_logic for s in stats},
                boards=[dict(seed=r[0].seed, **{n:g.outcome for n,g in zip(names,r)}) for r in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--train-start", type=int, default=710000)
    parser.add_argument("--validation-start", type=int, default=730000)
    parser.add_argument("--test-start", type=int, default=750000)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--validation-games", type=int, default=512)
    parser.add_argument("--validation-endgame", action="store_true")
    args = parser.parse_args()
    init_worker()
    src = Path("models/belief_rl.pt")
    steps = args.checkpoints
    if steps != sorted(set(steps)) or any(n <= 0 or n % 16 for n in steps):
        parser.error("checkpoints must be increasing positive multiples of 16")
    train = [args.train_start, args.train_start + steps[-1]]
    validation = [args.validation_start, args.validation_start + args.validation_games]
    test = [args.test_start, args.test_start + 1024]
    ranges = sorted([train, validation, test])
    if min(args.jobs, args.validation_games) < 1 or any(a[1] > b[0] for a,b in zip(ranges, ranges[1:])):
        parser.error("positive counts and disjoint seed ranges required")
    plan = dict(parent_sha256=hashlib.sha256(src.read_bytes()).hexdigest(), train=train,
                validation=validation, sealed_test=test, ranges="half-open", validation_endgame=args.validation_endgame,
                layouts=steps[-1], group=4, episodes=4*steps[-1], checkpoints=steps, lr=.002, batch=16, epochs=3,
                fresh_adam=True, rank="highest paired wins, earlier checkpoint breaks ties",
                validation_min_gain=.01, validation_max_p=.05/len(steps), final_max_p=.05,
                final_min_gain=.01, invalid_games_allowed=0, opening="random", cfg="16x30/99")
    path = args.out
    with path.open("x", encoding="utf-8") as out:
        json.dump(dict(stage="planned", plan=plan), out, indent=2)
    def save(report):
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        tmp.replace(path)
    base = torch.load(src, weights_only=True, map_location="cpu")
    actor = BeliefActor()
    actor.load_state_dict(base["model"])
    snapshots = {"live":cpu_weights(actor)}
    optimizer = torch.optim.Adam(actor.parameters(), lr=plan["lr"])
    batches = []
    with pool_for(args.jobs) as pool:
        dispatch = pool.map if pool else map
        for offset in range(0, plan["layouts"], plan["batch"]):
            start = plan["train"][0] + offset
            batches.append(train_batch(dispatch, actor, optimizer, range(start, start+plan["batch"]), plan["group"]))
            seen = offset + plan["batch"]
            if seen in steps:
                snapshots[str(seen)] = cpu_weights(actor)
            if seen % 64 == 0 or seen in steps:
                save(dict(stage="training", plan=plan, batches=batches))
                print(f"trained {seen} layouts", flush=True)
        names = list(snapshots) + (["endgame"] if args.validation_endgame else [])
        validation = compare(dispatch, range(*plan["validation"]), names, list(snapshots.values()), final=args.validation_endgame)
        chosen = max(map(str, steps), key=lambda k: validation["wins"][k])
        paired = next(p for p in validation["paired"] if p["challenger"] == chosen)
        qualifies = (not any(validation["invalid"].values()) and paired["win_rate_delta"] >= .01
                     and paired["p_value"] <= plan["validation_max_p"])
        report = dict(plan=plan, batches=batches, validation=validation, top_validation=chosen,
                      final_test_used=qualifies, candidate_passed=False)
        if qualifies:
            final = compare(dispatch, range(*plan["sealed_test"]), ["live",chosen,"endgame"],
                            [snapshots["live"],snapshots[chosen]], final=True)
            report["final"] = final
            pair = final["paired"][0]
            if not any(final["invalid"].values()) and pair["p_value"] <= .05 and pair["win_rate_delta"] >= .01:
                candidate = Path("models") / f"belief_candidate_{args.train_start}_{chosen}.pt"
                torch.save(dict(model=restore(BeliefActor(), snapshots[chosen]).state_dict(),
                                parent=plan["parent_sha256"], training=dict(seed_base=args.train_start,layouts=int(chosen))),
                           candidate)
                report.update(candidate_passed=True, candidate=str(candidate))
        save(dict(stage="complete", **report))
        print(chosen, validation["wins"], "final test used:", qualifies, flush=True)


if __name__ == "__main__":
    main()
