"""Offline evaluation: play N boards with a policy and report attributable numbers.

Protocol is fixed on purpose, because "win rate" is meaningless without one:
  * the same board set is replayed for every policy (paired comparison, McNemar test)
  * a board is generated once from a seed; no re-roll after a loss
  * every loss is labelled: "logic" (a safe move existed - a bug) or "guess" (a coin flip)
  * the per-game mine layout is regenerated from the seed, so runs are reproducible

Usage:
    python -m ms.eval --games 200 --policy exact
    python -m ms.eval --games 200 --compare exact,min_prob
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Iterable

import numpy as np

from .board import Board
from .engine import BEGINNER, EXPERT, INTERMEDIATE, Config, Minefield
from .opening import OPENINGS, first_cell
from .policy import (DeadGuessPolicy, DeepEndgamePolicy, EndgamePolicy, ExactSolverPolicy,
                     LearnedGuessPolicy, LookaheadPolicy, MidgamePolicy, MinProbPolicy, RolloutPolicy)
from .solver import certain_moves, Inconsistent, raw_constraints

POLICIES = {
    "exact": ExactSolverPolicy,
    "lookahead": LookaheadPolicy,
    "endgame": EndgamePolicy,
    "deep_endgame": DeepEndgamePolicy,
    "midgame": MidgamePolicy,
    "rollout": RolloutPolicy,
    "deadguess": DeadGuessPolicy,
    "min_prob": MinProbPolicy,
    "learned": LearnedGuessPolicy,
}
MODEL_WEIGHTS = "models/guess_scorer.pt"

MAX_MOVES = 4000  # a 16x30 board needs < 500 openings; more means something is looping


@dataclass
class GameResult:
    seed: int
    won: bool
    moves: int
    outcome: str            # "win" | "logic" | "guess" | "stuck"
    wall_ms: float = 0.0
    degraded: int = 0       # decisions that had to fall back to sampling
    endgame_solved: int = 0
    endgame_switches: int = 0
    rollout_comparisons: int = 0
    rollout_switches: int = 0
    deadguess_found: int = 0
    deadguess_switches: int = 0


@dataclass
class GameStats:
    policy: str
    cfg: dict
    games: int = 0
    wins: int = 0
    losses_logic: int = 0
    losses_guess: int = 0
    stuck: int = 0
    moves_total: int = 0
    wall_s: float = 0.0
    degraded: int = 0
    results: list[GameResult] = field(default_factory=list)
    elapsed_s: float = 0.0   # batch wall time, including parallel worker overhead
    endgame_reached_games: int = 0
    endgame_changed_games: int = 0
    endgame_solved: int = 0
    endgame_switches: int = 0
    rollout_games: int = 0
    rollout_changed_games: int = 0
    rollout_comparisons: int = 0
    rollout_switches: int = 0
    deadguess_found: int = 0
    deadguess_switches: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def ci95(self) -> tuple[float, float]:
        """Wilson score interval - honest for proportions near 0 or 1."""
        n, p = self.games, self.win_rate
        if n == 0:
            return (0.0, 1.0)
        z = 1.959963985
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return (max(0.0, centre - half), min(1.0, centre + half))

    def summary(self) -> dict:
        lo, hi = self.ci95
        return {
            "policy": self.policy,
            "games": self.games,
            "wins": self.wins,
            "elapsed_s": round(self.elapsed_s, 3),
            "win_rate": round(self.win_rate, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "losses_logic": self.losses_logic,
            "losses_guess": self.losses_guess,
            "stuck": self.stuck,
            "mean_moves": round(self.moves_total / self.games, 2) if self.games else 0,
            "mean_ms_per_move": round(1000 * self.wall_s / self.moves_total, 2)
            if self.moves_total else 0,
            "degraded": self.degraded,
            "cfg": self.cfg,
            **({"endgame_reached_games": self.endgame_reached_games,
                "endgame_changed_games": self.endgame_changed_games,
                "endgame_solved": self.endgame_solved,
                "endgame_switches": self.endgame_switches}
               if self.policy in ("endgame", "deep_endgame", "midgame", "rollout", "deadguess")
               else {}),
            **({"deadguess_found": self.deadguess_found, "deadguess_switches": self.deadguess_switches}
               if self.policy == "deadguess" else {}),
            **({"rollout_games": self.rollout_games,
                "rollout_changed_games": self.rollout_changed_games,
                "rollout_comparisons": self.rollout_comparisons,
                "rollout_switches": self.rollout_switches} if self.policy == "rollout" else {}),
        }


def play_game(seed: int, policy, cfg: Config, max_moves: int = MAX_MOVES, *,
              opening: str = 'random') -> GameResult:
    """One game, no re-rolls. Losses are labelled by whether a safe move was available."""
    rng = np.random.default_rng(seed * 1_000_003 + 7)
    mf = Minefield(cfg, rng)
    moves = 0
    degraded = solved = switches = 0
    rollout_comparisons = rollout_switches = deadguess_found = deadguess_switches = 0
    t0 = time.perf_counter()

    def finish(outcome: str) -> GameResult:
        return GameResult(seed, mf.won, moves, outcome, (time.perf_counter() - t0) * 1000,
                          degraded, solved, switches, rollout_comparisons, rollout_switches,
                          deadguess_found, deadguess_switches)

    res = mf.click(first_cell(cfg, seed, opening))
    moves += 1
    if res == "boom":
        return GameResult(seed, False, moves, "guess", 0.0, 0)

    while moves < max_moves:
        if mf.won:
            break
        board = mf.board()
        # was a zero-risk move available before the policy committed?
        raw, _, _ = raw_constraints(board)
        safe, mines = certain_moves(raw)
        move = policy.act(board)
        step = getattr(getattr(policy, "solver", None), "last_stats", {})
        degraded += bool(step.get("degraded"))
        solved += bool(step.get("endgame_solved"))
        switches += bool(step.get("endgame_switch"))
        rollout_comparisons += bool(step.get("rollout_solved"))
        rollout_switches += bool(step.get("rollout_switch"))
        deadguess_found += bool(step.get("deadguess_found"))
        deadguess_switches += bool(step.get("deadguess_switch"))
        if move.kind == "done":
            return finish("win" if mf.won else "stuck")
        if move.kind == "flag":
            for (r, c) in move.cells:
                mf.flag(r * cfg.cols + c)
            moves += 1
            continue
        # opening (or guessing) one cell at a time keeps the loss attribution exact
        r, c = move.cells[0]
        res = mf.click(r * cfg.cols + c)
        moves += 1
        if res == "boom":
            # a loss is a logic bug iff a zero-risk move existed and was not taken
            outcome = "logic" if safe else "guess"
            return finish(outcome)
    return finish("win" if mf.won else "stuck")


def _make_policy(name: str, seed: int):
    cls = POLICIES[name]
    if cls is LearnedGuessPolicy:
        return cls(seed=seed, weights=MODEL_WEIGHTS)
    return cls(seed=seed)


def _one(args):
    policy_name, seed, cfg, opening = args
    return play_game(seed, _make_policy(policy_name, seed), cfg, opening=opening)


def run(policy_name: str, games: int, cfg: Config, seed_base: int = 0,
        jobs: int = 1, *, opening: str = 'random') -> GameStats:
    """Replay the requested seeds; each game starts with a fresh policy and RNG."""
    work = [(policy_name, seed_base + i, cfg, opening) for i in range(games)]
    start = time.perf_counter()
    if jobs > 1:
        import multiprocessing as mp

        with mp.Pool(jobs) as pool:
            results = pool.map(_one, work, chunksize=max(1, games // (jobs * 8)))
    elif policy_name == "learned":
        # Reuse the checkpoint in serial runs, not the previous game's solver RNG.
        policy = _make_policy(policy_name, seed_base)
        results = [play_game(seed, LearnedGuessPolicy(seed=seed, model=policy.model), cfg,
                             opening=opening)
                   for _, seed, _, opening in work]
    else:
        results = [_one(args) for args in work]
    return summarize(policy_name, cfg, results, time.perf_counter() - start)


def summarize(policy_name: str, cfg: Config, results: Iterable[GameResult],
              elapsed_s: float) -> GameStats:
    stats = GameStats(policy=policy_name, cfg=asdict(cfg), elapsed_s=elapsed_s)
    for res in results:
        stats.games += 1
        stats.moves_total += res.moves
        stats.wall_s += res.wall_ms / 1000
        stats.degraded += res.degraded
        stats.endgame_reached_games += bool(res.endgame_solved)
        stats.endgame_changed_games += bool(res.endgame_switches)
        stats.endgame_solved += res.endgame_solved
        stats.endgame_switches += res.endgame_switches
        stats.rollout_games += bool(res.rollout_comparisons)
        stats.rollout_changed_games += bool(res.rollout_switches)
        stats.rollout_comparisons += res.rollout_comparisons
        stats.rollout_switches += res.rollout_switches
        stats.deadguess_found += res.deadguess_found
        stats.deadguess_switches += res.deadguess_switches
        if res.won:
            stats.wins += 1
        elif res.outcome == "logic":
            stats.losses_logic += 1
        elif res.outcome == "guess":
            stats.losses_guess += 1
        else:
            stats.stuck += 1
        stats.results.append(res)
    return stats


def mcnemar(a: GameStats, b: GameStats) -> dict:
    """Paired comparison on identical board seeds: how often does one win where the other
    loses? Uses the exact binomial form, which is valid for small discordant counts."""
    by_seed_a = {r.seed: r.won for r in a.results}
    by_seed_b = {r.seed: r.won for r in b.results}
    common = sorted(set(by_seed_a) & set(by_seed_b))
    n01 = sum(1 for s in common if not by_seed_a[s] and by_seed_b[s])
    n10 = sum(1 for s in common if by_seed_a[s] and not by_seed_b[s])
    n = n01 + n10
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n) * 2 if n else 1.0
    pairs = len(common)
    delta = (n01 - n10) / pairs if pairs else None
    interval = None
    if pairs and n:
        # A bootstrap of matched board outcomes {-1, 0, +1} can be drawn directly
        # from these three counts. No discordant pairs => don't report a [0, 0] CI.
        rng = np.random.default_rng(0)
        draws = rng.multinomial(pairs, (n10 / pairs, (pairs - n) / pairs, n01 / pairs),
                                size=20_000)
        boot = (draws[:, 2] - draws[:, 0]) / pairs
        interval = [round(float(x), 4) for x in np.quantile(boot, (0.025, 0.975))]
    return {
        "pairs": pairs,
        "baseline": a.policy,
        "challenger": b.policy,
        f"{a.policy}_only": n10,
        f"{b.policy}_only": n01,
        "win_rate_delta": round(delta, 4) if delta is not None else None,
        "win_rate_delta_ci95_paired_bootstrap": interval,
        "p_value": round(min(1.0, p), 6),
    }


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--preset", default="", choices=["", "beginner", "intermediate", "expert"])
    ap.add_argument("--policy", default="exact", choices=POLICIES)
    compare = ap.add_mutually_exclusive_group()
    compare.add_argument("--compare", default="", help="comma-separated policies; first is the baseline")
    compare.add_argument("--compare-openings", default=None,
                         help="comma-separated opening rules, holding --policy constant")
    ap.add_argument("--opening", choices=OPENINGS, default=None,
                    help="first-click rule (default: random)")
    ap.add_argument("--rows", type=int, default=INTERMEDIATE.rows)
    ap.add_argument("--cols", type=int, default=INTERMEDIATE.cols)
    ap.add_argument("--mines", type=int, default=INTERMEDIATE.mines)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1, help="parallel games (1 = single process)")
    ap.add_argument("--out", default="", help="write the JSON report here")
    args = ap.parse_args(list(argv) if argv is not None else None)

    presets = {
        "beginner": BEGINNER,
        "intermediate": INTERMEDIATE,
        "expert": EXPERT,
    }
    cfg = presets[args.preset] if args.preset else Config(args.rows, args.cols, args.mines)
    report: dict = {"cfg": asdict(cfg), "games": args.games, "seed_base": args.seed_base,
                    "jobs": args.jobs}

    if args.compare_openings is not None:
        names = [n.strip() for n in args.compare_openings.split(',')]
        if args.opening is not None:
            ap.error('--opening cannot be used with --compare-openings')
        if len(names) < 2 or len(set(names)) != len(names) or any(n not in OPENINGS for n in names):
            ap.error(f"--compare-openings needs at least two distinct rules from {', '.join(OPENINGS)}")
        report.update(policy=args.policy,
                      comparison='shared seeds; mine layouts may differ across opening rules',
                      openings={})
        for name in names:
            stats = run(args.policy, args.games, cfg, args.seed_base, args.jobs, opening=name)
            picks = {first_cell(cfg, seed, name)
                     for seed in range(args.seed_base, args.seed_base + args.games)}
            result = dict(stats.summary(), opening=name, unique_first_cells=len(picks))
            report['openings'][name] = result
            print(json.dumps(result, ensure_ascii=False), flush=True)
    elif args.compare:
        opening = report['opening'] = args.opening or 'random'
        names = [n.strip() for n in args.compare.split(",") if n.strip()]
        if len(names) < 2 or len(set(names)) != len(names) or any(n not in POLICIES for n in names):
            ap.error(f"--compare needs at least two distinct policies from {', '.join(POLICIES)}")
        stats = {n: run(n, args.games, cfg, args.seed_base, args.jobs, opening=opening)
                 for n in names}
        report["policies"] = {n: s.summary() for n, s in stats.items()}
        report["paired"] = [mcnemar(stats[names[0]], stats[n]) for n in names[1:]]
        for line in report["policies"].values():
            print(json.dumps(line, ensure_ascii=False))
        for line in report["paired"]:
            print(json.dumps(line, ensure_ascii=False))
    else:
        opening = report['opening'] = args.opening or 'random'
        s = run(args.policy, args.games, cfg, args.seed_base, args.jobs, opening=opening)
        report["policies"] = {s.policy: s.summary()}
        print(json.dumps(s.summary(), ensure_ascii=False))

    if args.out:
        from pathlib import Path

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
