"""Evaluate model guesses that change only after exact comparison of terminal values."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from experiments.belief_rl import init_worker
from experiments.guess_patterns import CountGuesses, guess_survival_delta
from ms.endgame import _worlds
from ms.engine import EXPERT, Minefield
from ms.eval import GameResult, mcnemar, play_game, summarize
from ms.opening import first_cell
from ms.play_web import load_policy, WEIGHTS
from ms.policy import EndgamePolicy
from ms.rollout import _finish
from ms.solver import Move
from ms.verified_search import Limits, VerifiedActor


def pair(seed):
    live = load_policy(seed, review=False)
    verified = VerifiedActor(live.actor, seed)
    row = dict(seed=seed, first_cell=first_cell(EXPERT, seed, "random"))
    for name, policy in (("live", live), ("verified", verified), ("endgame", EndgamePolicy(seed))):
        counted = CountGuesses(policy)
        result = play_game(seed, counted, EXPERT, opening="random")
        if result.outcome not in ("win", "guess") or (result.outcome == "guess" and not counted.count):
            raise RuntimeError(f"invalid result for {name} on {seed}: {result.outcome}")
        row[name] = dict(won=result.won, guesses=counted.count, survived=counted.count-int(not result.won))
    row.update(changes=verified.changes, eligible=verified.eligible, checked=verified.checked,
               complete=verified.complete, extra_seconds=round(verified.extra_seconds, 6))
    return row


def audit_loss(row):
    """Recover a historical pre-click board and replay every public completion."""
    seed = row["seed"]
    field = Minefield(EXPERT, np.random.default_rng(seed*1_000_003 + 7))
    field.click(first_cell(EXPERT, seed, "random"))
    live = load_policy(seed, review=False)
    verified = VerifiedActor(live.actor, seed)
    while not field.won:
        board = field.board()
        move = live.act(board)
        if move.kind == "flag":
            for r, c in move.cells:
                field.flag(r*board.cols+c)
            continue
        r, c = move.cells[0]
        first = r*board.cols+c
        if field.click(first) == "boom":
            break
    if field.won or first != row["death"]["cell"] or move.kind != "guess":
        raise RuntimeError(f"could not reproduce historical guess on {seed}")
    alternative = verified.act(board)
    r, c = alternative.cells[0]
    chosen = r*board.cols+c
    hidden = tuple(map(int, np.flatnonzero(board.covered())))
    worlds = _worlds(board, hidden, live.solver.last_stats["solutions"], Limits().enumeration)
    caches = {"live": {}, "verified": {}}

    def follow(policy, cache, visible):
        key = visible.cells.tobytes()
        if key not in cache:
            action = policy.act(visible)
            # Match play_game: one open/guess, all flags.
            cache[key] = (Move(action.kind, action.cells[:1], action.p_mine, action.reason)
                          if action.kind == "open" else action)
        return cache[key]

    change = verified.changes[0].copy() if verified.changes else None
    results = Counter()
    for mask in worlds:
        layout = board.flags().ravel().copy()
        for i, cell in enumerate(hidden):
            layout[cell] = bool(mask & (1 << i))
        field.load(layout)
        for tag, initial, policy, cache in (("live", first, live, caches["live"]),
                                            ("new_first_only", chosen, live, caches["live"]),
                                            ("verified", chosen, verified, caches["verified"])):
            field.visible[:] = board.cells.ravel()
            field.won = field.exploded = False
            outcome = _finish(field, initial, lambda b: follow(policy, cache, b))
            if outcome is None:
                raise RuntimeError(f"incomplete continuation on {seed}")
            results[tag] += int(outcome)
    return dict(seed=seed, cells=board.cells.tolist(), remaining_mines=board.mines_left,
                worlds=len(worlds), change=change, wins_by_real_policy=dict(results))


def summarize_rows(rows):
    stats, metrics = {}, {}
    for name in ("live", "verified", "endgame"):
        results = [GameResult(r["seed"], r[name]["won"], 0, "win" if r[name]["won"] else "guess") for r in rows]
        stats[name] = summarize(name, EXPERT, results, 0)
        guesses = sum(r[name]["guesses"] for r in rows)
        safe = sum(r[name]["survived"] for r in rows)
        metrics[name] = dict(wins=stats[name].wins, games=len(rows), guesses=guesses, survived=safe,
                             win_rate=stats[name].win_rate, guess_survival=safe/guesses if guesses else None)
    return dict(metrics=metrics, paired_wins=[mcnemar(stats[n], stats["verified"]) for n in ("live", "endgame")],
                paired_guess_survival=guess_survival_delta(rows, "live", "verified"),
                changed_boards=sum(bool(r["changes"]) for r in rows), changes=sum(len(r["changes"]) for r in rows),
                checked=sum(r["checked"] for r in rows), eligible=sum(r["eligible"] for r in rows),
                fully_searched=sum(r["complete"] for r in rows), boards=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed-base", type=int, default=3100000)
    p.add_argument("--games", type=int, default=2048)
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--out", type=Path, default=Path("reports/verified_guesses_3100000.json"))
    p.add_argument("--audit-out", type=Path, help="Only replay historical positions with a known optimality gap")
    args = p.parse_args()
    if args.jobs < 1:
        p.error("positive worker count required")
    init_worker()
    parent = torch.load(WEIGHTS, map_location="cpu", weights_only=True)["training"]
    if args.audit_out:
        source = Path("reports/web_guess_holdout_1024.json")
        original = json.loads(source.read_text(encoding="utf8"))
        if original["weights_sha256"] != sha256(WEIGHTS.read_bytes()).hexdigest():
            p.error("historical report uses different weights")
        cases = [r for r in original["boards"] if (r.get("death", {}).get("optimal_win_gap") or 0) > 0]
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=init_worker) as pool:
            audited = list(pool.map(audit_loss, cases))
        args.audit_out.parent.mkdir(parents=True, exist_ok=True)
        args.audit_out.write_text(json.dumps(dict(source=source.name, source_sha256=sha256(source.read_bytes()).hexdigest(),
                                                   limits=asdict(Limits()), cases=audited), indent=2)+"\n", encoding="utf8")
        return
    end = args.seed_base + args.games
    overlaps_training = max(args.seed_base, parent["seed_base"]) < min(end, parent["seed_base"] + parent["layouts"])
    if args.games < 1 or args.seed_base < 0 or overlaps_training:
        p.error("require positive counts and seeds independent of training")
    plan = dict(seed_base=args.seed_base, games=args.games, opening="random", cfg=asdict(EXPERT),
                limits=asdict(Limits()), weights_sha256=sha256(WEIGHTS.read_bytes()).hexdigest())
    with ProcessPoolExecutor(max_workers=args.jobs, initializer=init_worker) as pool:
        rows = []
        for row in pool.map(pair, range(args.seed_base, end), chunksize=2):
            rows.append(row)
            if len(rows) % 256 == 0:
                print(f"{len(rows)}/{args.games}", flush=True)
    result = summarize_rows(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(plan=plan, **result), indent=2)+"\n", encoding="utf8")
    print({k:v for k,v in result.items() if k!="boards"}, flush=True)


if __name__ == "__main__":
    main()
