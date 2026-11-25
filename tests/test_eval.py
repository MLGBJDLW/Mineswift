"""The evaluator must report honest, attributable numbers."""
from __future__ import annotations

import json

from ms.engine import BEGINNER
from ms.eval import GameResult, POLICIES, main, mcnemar, play_game, summarize


def test_exact_policy_never_loses_by_logic():
    """A loss labelled 'logic' means a safe move existed and was skipped - a bug."""
    for seed in range(12):
        res = play_game(seed, POLICIES["exact"](seed=0), BEGINNER)
        assert res.outcome in ("win", "guess"), res


def test_same_seed_same_game():
    a = play_game(3, POLICIES["exact"](seed=0), BEGINNER)
    b = play_game(3, POLICIES["exact"](seed=0), BEGINNER)
    assert (a.won, a.moves, a.outcome) == (b.won, b.moves, b.outcome)


def test_min_prob_baseline_is_weaker_but_sane():
    """The heuristic baseline must still play: it opens cells and eventually dies or wins."""
    stats = [play_game(s, POLICIES["min_prob"](seed=0), BEGINNER) for s in range(10)]
    assert all(r.moves > 1 for r in stats)
    assert all(r.outcome in ("win", "guess") for r in stats)


def test_comparison_reports_delta_on_shared_seeds_only():
    def stats(name, wins):
        return summarize(name, BEGINNER,
                         [GameResult(seed, won, 1, "win" if won else "guess")
                          for seed, won in wins.items()], 0.0)

    exact = stats("exact", {1: True, 2: False, 3: False, 4: True, 5: True})
    ahead = stats("lookahead", {1: False, 2: True, 3: True, 4: True, 6: False})
    pair = mcnemar(exact, ahead)
    assert pair["pairs"] == 4
    assert pair["baseline"] == "exact" and pair["challenger"] == "lookahead"
    assert (pair["exact_only"], pair["lookahead_only"]) == (1, 2)
    assert pair["win_rate_delta"] == 0.25
    lo, hi = pair["win_rate_delta_ci95_paired_bootstrap"]
    assert lo < 0.25 < hi


def test_cli_can_write_lookahead_comparison(tmp_path):
    path = tmp_path / "paired.json"
    assert main(["--preset", "beginner", "--games", "2", "--seed-base", "5000",
                 "--compare", "exact,lookahead", "--out", str(path)]) == 0
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["games"] == 2 and report["seed_base"] == 5000 and report["jobs"] == 1
    assert report["paired"][0]["pairs"] == 2
    assert all(p["elapsed_s"] >= 0 for p in report["policies"].values())
