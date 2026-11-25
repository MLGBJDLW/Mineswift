"""Paired terminal-reward training with extra emphasis on public guess patterns."""
import argparse
from collections import Counter
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from experiments.belief_rl import collect, cpu_weights, init_worker, optimize, pool_for, restore
from ms.belief_rl import BeliefActor, Observation
from ms.belief_search import RolloutActor
from ms.board import COVERED, FLAG
from ms.engine import EXPERT, neighbour_ids
from ms.eval import GameResult, mcnemar, play_game, summarize
from ms.opening import first_cell
from ms.play_web import WEIGHTS
from ms.policy import EndgamePolicy


@lru_cache(maxsize=8)
def neighbors(shape):
    return neighbour_ids(*shape)


def pattern(obs):
    """Name a state before the action, using only the actor's public information."""
    covered = obs.cells == COVERED
    risk = obs.planes[0]
    safest = covered & (risk <= risk[covered].min() + 1e-7)
    if safest.sum() > 1 and np.all(obs.planes[2][safest] == 1):
        if np.ptp(obs.planes[3][safest]) >= .25:
            return "equal_risk_different_information"
    adj = neighbors(obs.cells.shape)
    cells = np.append(obs.cells, -4)[adj]
    unknown = cells == COVERED
    degree = unknown.sum(axis=1)
    needed = obs.cells.ravel() - (cells == FLAG).sum(axis=1)
    clues = (needed == 1) & ((degree == 2) | (degree == 3))
    if clues.any():
        local = np.append(risk, 0)[adj[clues]]
        mask = unknown[clues]
        spread = np.where(mask, local, -np.inf).max(1) - np.where(mask, local, np.inf).min(1)
        if np.any(spread >= .1 - 1e-7):
            return "one_mine_unequal_risk"
    return "other"


class MeasuredActor(RolloutActor):
    def position(self, board):
        state = super().position(board)
        if self.guesses == 0 and isinstance(state, Observation):
            self.first_pattern = pattern(state)
        return state


class CountGuesses:
    def __init__(self, policy):
        self.policy = policy
        self.solver = policy.solver
        self.count = 0
        self.first_board = None

    def act(self, board):
        move = self.policy.act(board)
        if move.kind == "guess":
            if not self.count:
                self.first_board = sha256(board.cells.tobytes()).hexdigest()
            self.count += 1
        return move


def evaluate_seed(task):
    seed, weights = task
    row = dict(seed=seed, first_cell=first_cell(EXPERT, seed, "random"))
    policies = [(name, MeasuredActor(restore(BeliefActor(), w), seed)) for name, w in weights.items()]
    policies.append(("endgame", EndgamePolicy(seed)))
    for name, policy in policies:
        counter = CountGuesses(policy)
        result = play_game(seed, counter, EXPERT, opening="random")
        if result.outcome not in ("win", "guess") or (result.outcome == "guess" and not counter.count):
            raise RuntimeError(f"invalid game: {name}, {seed}, {result.outcome}")
        row[name] = dict(won=result.won, guesses=counter.count,
                         survived=counter.count - int(not result.won), first_board=counter.first_board)
        if name == "live":
            row["first_pattern"] = getattr(policy, "first_pattern", None)
    return row


def guess_survival_delta(rows, baseline, challenger):
    """Resample whole paired boards so guesses from one trajectory stay together."""
    counts = np.array([[row[n][k] for n in (baseline, challenger) for k in ("survived", "guesses")]
                       for row in rows])
    safe_a, total_a, safe_b, total_b = counts.sum(axis=0)
    if not total_a or not total_b:
        return None
    delta = safe_b/total_b - safe_a/total_a
    rng = np.random.default_rng(1)
    draws = []
    for _ in range(100):
        ids = rng.integers(len(rows), size=(100, len(rows)))
        a, na, b, nb = counts[ids].sum(axis=1).T
        valid = (na > 0) & (nb > 0)
        draws.extend(b[valid]/nb[valid] - a[valid]/na[valid])
    return dict(baseline=baseline, challenger=challenger, delta=delta,
                ci95=np.quantile(draws, [.025, .975]).tolist())


def summary(rows, names):
    wins, first, metrics = {}, {}, {}
    for name in names:
        records = [row[name] for row in rows]
        shots = sum(r["guesses"] for r in records)
        survived = sum(r["survived"] for r in records)
        wins[name] = summarize(name, EXPERT, [GameResult(row["seed"], row[name]["won"], 0,
                                "win" if row[name]["won"] else "guess") for row in rows], 0)
        first_results = []
        by_pattern = {}
        for row, rec in zip(rows, records):
            if rec["first_board"] and rec["first_board"] == row["live"]["first_board"]:
                safe = rec["survived"] > 0
                first_results.append(GameResult(row["seed"], safe, 0, "win" if safe else "guess"))
                group = by_pattern.setdefault(row["first_pattern"], dict(boards=0, safe=0))
                group["boards"] += 1
                group["safe"] += int(safe)
        first[name] = summarize(name, EXPERT, first_results, 0)
        metrics[name] = dict(wins=wins[name].wins, games=len(rows), win_ci95=wins[name].ci95,
                             guesses=shots, safe_guesses=survived, guess_survival=survived/shots if shots else None,
                             matched_first_guesses=len(first_results), safe_first_guesses=first[name].wins,
                             first_by_pattern=by_pattern)
    comparisons = [mcnemar(wins[name], wins["patterns"]) for name in names if name != "patterns"]
    return dict(metrics=metrics, paired_wins=comparisons,
                paired_guess_survival=[guess_survival_delta(rows, name, "patterns")
                                       for name in ("live", "ordinary")],
                paired_first_guess=[mcnemar(first[name], first["patterns"]) for name in names if name != "patterns"])


def train_arm(dispatch, actor, start, layouts, focus):
    optimizer = torch.optim.Adam(actor.parameters(), lr=.002)
    history = []
    for offset in range(0, layouts, 16):
        n = min(16, layouts-offset)
        weights = cpu_weights(actor)
        tasks = [(weights, seed, 4, 30, False, False) for seed in range(start+offset, start+offset+n)]
        groups = list(dispatch(collect, tasks))
        data = [sample for _, _, samples in groups for sample in samples]
        kinds = [pattern(d.obs) for d, _ in data]
        if focus and data:
            priority = np.array([1. if k == "other" else 3. for k in kinds])
            data = [(d, a*w) for (d, a), w in zip(data, priority/priority.mean())]
        diagnostics = []
        optimize(actor, optimizer, data, n*4, 3, diagnostics)
        history.append(dict(layouts=n, wins=sum(w for w, _, _ in groups),
                            guesses=sum(g for _, g, _ in groups), nonzero_advantage_samples=dict(Counter(kinds)),
                            epochs=diagnostics))
        if (offset+n) % 128 == 0:
            print("patterns" if focus else "ordinary", offset+n, flush=True)
    return history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layouts", type=int, default=512)
    p.add_argument("--games", type=int, default=1024)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--train-start", type=int, default=2610000)
    p.add_argument("--eval-start", type=int, default=2710000)
    p.add_argument("--final-start", type=int, default=2810000)
    p.add_argument("--out", type=Path, default=Path("reports/guess_pattern_trial.json"))
    p.add_argument("--weights-dir", type=Path, default=Path("reports/local/guess_patterns"))
    args = p.parse_args()
    init_worker()
    saved = torch.load(WEIGHTS, weights_only=True, map_location="cpu")
    parent = saved["training"]
    ranges = sorted((s, s+n) for s,n in ((args.train_start,args.layouts),(args.eval_start,args.games),
                    (args.final_start,args.games),(parent["seed_base"],parent["layouts"])))
    if min(args.layouts, args.games, args.jobs) < 1 or ranges[0][0] < 0 or any(a[1]>b[0] for a,b in zip(ranges,ranges[1:])):
        p.error("positive counts and disjoint board seeds required")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    plan = dict(parent_sha256=sha256(WEIGHTS.read_bytes()).hexdigest(), train_start=args.train_start,
                eval_start=args.eval_start, final_start=args.final_start, games_per_eval=args.games,
                layouts_per_arm=args.layouts, trajectories_per_layout=4, batch=16, epochs=3, lr=.002,
                pattern_weight=3, opening="random", difficulty="expert", reward="terminal +/-1",
                guess_survival_interval="secondary, 10000 bootstrap draws of paired boards",
                gate="paired wins vs both live and ordinary: delta >= .01 and exact p <= .05, then replicate")
    names = ["live","ordinary","patterns","endgame"]
    report = dict(plan=plan, training={}, checkpoint_sha256={}, final_test_used=False)
    weights = {}
    with pool_for(args.jobs) as pool:
        dispatch = pool.map if pool else map
        for name in names[:3]:
            actor = BeliefActor().eval()
            actor.load_state_dict(saved["model"])
            if name != "live":
                report["training"][name] = train_arm(dispatch, actor, args.train_start, args.layouts, name=="patterns")
                path = args.weights_dir / (name + ".pt")
                torch.save(dict(model=actor.state_dict(), plan=plan, arm=name), path)
                report["checkpoint_sha256"][name] = sha256(path.read_bytes()).hexdigest()
            weights[name] = cpu_weights(actor)
        for stage, start in (("evaluation",args.eval_start),("replication",args.final_start)):
            tasks = [(s,weights) for s in range(start,start+args.games)]
            rows = []
            for row in (pool.imap(evaluate_seed,tasks,chunksize=2) if pool else map(evaluate_seed,tasks)):
                rows.append(row)
                if len(rows) % 128 == 0:
                    print(stage, len(rows), flush=True)
            result = summary(rows,names)
            report[stage] = dict(**result, boards=rows)
            passed = all(q["win_rate_delta"]>=.01 and q["p_value"]<=.05
                         for q in result["paired_wins"] if q["baseline"] in ("live","ordinary"))
            if stage=="evaluation":
                report["final_test_used"] = passed
            print(stage, result["metrics"], flush=True)
            if not passed:
                break
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf8")


if __name__ == "__main__":
    main()
