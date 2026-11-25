"""Decision policies. This is where the learned part of the agent lives.

Every policy takes a Board and returns a Move, so policies are interchangeable in
evaluation and the learned one is compared against the exact solver on the same boards.

    ExactSolverPolicy  - deterministic solver (reference)
    LookaheadPolicy    - same solver, with one-reveal lookahead on exact risk ties
    MinProbPolicy      - classic heuristic: open the cell with the lowest local mine ratio
                         (no global mine-count reasoning). Included as the honest baseline
                         everyone actually ships, to show what the solver buys.

The learned scorer (LearnedGuessPolicy) is wired in model.py and only ever decides the
cases where no logical move exists, so it can never make the agent play worse in logic.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .board import COVERED, FLAG, QUESTION, Board
from .deadguess import forced_pair
from .endgame import solve_endgame
from .lookahead import OneStepRanker
from .midgame import MidgameRanker
from .rollout import compare_rollouts
from .solver import Cell, Move, Solver, raw_constraints

NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _nbrs(rows: int, cols: int, r: int, c: int):
    for dr, dc in NEIGHBOURS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            yield rr, cc


class ExactSolverPolicy:
    """The deterministic solver: exact global probabilities, zero-risk moves first."""

    name = "exact"

    def __init__(self, seed: int = 0):
        self.solver = Solver(seed=seed)

    def act(self, board: Board) -> Move:
        return self.solver.decide(board)


class LookaheadPolicy(ExactSolverPolicy):
    """The reference solver, with an opt-in tie-breaker on unproven opens."""

    name = "lookahead"

    def __init__(self, seed: int = 0):
        super().__init__(seed=seed)
        self.ranker = OneStepRanker()

    def act(self, board: Board) -> Move:
        return self.solver.decide(board, guess_ranker=self.ranker)


class EndgamePolicy(ExactSolverPolicy):
    """Keep proven moves; when small enough, maximize the chance of winning the game."""

    name = "endgame"

    def _baseline(self, board: Board) -> Move:
        return super().act(board)

    def _search(self, board: Board, cell: Cell, worlds: int):
        return solve_endgame(board, cell, worlds)

    def act(self, board: Board) -> Move:
        move = self._baseline(board)
        stats = self.solver.last_stats
        if move.kind != "guess" or stats.get("degraded"):
            return move
        result = self._search(board, move.cells[0], stats["solutions"])
        if result is None:
            return move
        stats["endgame_solved"] = True
        if result.cell == move.cells[0]:
            return move
        stats["endgame_switch"] = True
        return Move("guess", (result.cell,), result.mine_worlds / result.worlds,
                    f"endgame: optimal win chance {result.p_win:.4f} ({result.worlds} layouts)")


class DeepEndgamePolicy(EndgamePolicy):
    """Use a larger but deterministic work budget for exact game-tree search."""

    name = "deep_endgame"

    def _search(self, board: Board, cell: Cell, worlds: int):
        return solve_endgame(board, cell, worlds, max_cells=64, max_worlds=1000,
                             max_nodes=24_000, max_work=300_000)


class MidgamePolicy(EndgamePolicy):
    """Keep exact endgame search; compare near-risk guesses everywhere else."""

    name = "midgame"

    def __init__(self, seed: int = 0):
        super().__init__(seed=seed)
        self.ranker = MidgameRanker()
        self._guess_counts: dict[Cell, int] = {}

    def _baseline(self, board: Board) -> Move:
        return self.solver.decide(board, exact_guess_counts=self._guess_counts)

    def act(self, board: Board) -> Move:
        move = super().act(board)
        stats = self.solver.last_stats
        if move.kind != "guess" or stats.get("endgame_solved") or not self._guess_counts:
            return move
        worlds = stats["solutions"]
        result = self.ranker.choose(board, move.cells[0], self._guess_counts, worlds)
        if result is None or result[0] == move.cells[0]:
            return move
        cell, value = result
        risk = self._guess_counts[cell] / worlds
        stats.update(midgame_switch=True, guess_p=risk)
        return Move("guess", (cell,), risk,
                    f"midgame: two-click safety >= {value.two_click_survivors / worlds:.4f}")


class DeadGuessPolicy(EndgamePolicy):
    """Resolve certified 50/50s early, otherwise behave like endgame."""

    name = 'deadguess'

    def act(self, board: Board) -> Move:
        move = super().act(board)
        stats = self.solver.last_stats
        if move.kind != 'guess' or stats.get('endgame_solved') or stats.get('degraded'):
            return move
        pair = forced_pair(board)
        if pair is None:
            return move
        stats['deadguess_found'] = True
        if pair[0] == move.cells[0]:
            return move
        stats.update(deadguess_switch=True, guess_p=0.5)
        return Move('guess', (pair[0],), 0.5, 'certified symmetric one-mine pair')


class RolloutPolicy(EndgamePolicy):
    """Consider terminal wins on sampled layouts when exact endgame search cannot finish."""

    name = "rollout"

    def __init__(self, seed: int = 0):
        super().__init__(seed=seed)
        self.ranker = MidgameRanker(max_candidates=3)
        self._guess_counts: dict[Cell, int] = {}

    def _baseline(self, board: Board) -> Move:
        return self.solver.decide(board, exact_guess_counts=self._guess_counts)

    @staticmethod
    def _continuation(number: int):
        # No recursive rollout: use the proven baseline (plus exact endgames).
        return EndgamePolicy(seed=number).act

    def act(self, board: Board) -> Move:
        move = super().act(board)
        stats = self.solver.last_stats
        if (move.kind != 'guess' or stats.get('endgame_solved')
                or not 2 <= len(self._guess_counts) <= 150):
            return move
        worlds = stats['solutions']
        candidates = self.ranker.shortlist(board, move.cells[0], self._guess_counts, worlds)
        result = compare_rollouts(board, candidates, self._continuation)
        if result is None:
            return move
        stats['rollout_solved'] = True
        if result.cell == move.cells[0]:
            return move
        risk = self._guess_counts[result.cell] / worlds
        stats.update(rollout_switch=True, guess_p=risk)
        return Move('guess', (result.cell,), risk,
                    f'rollout: {result.wins}/{result.draws} wins vs reference {result.baseline_wins}')


class MinProbPolicy:
    """Baseline: local ratio heuristic, the way most open-source bots play.

    For each frontier cell it averages the "remaining mines / covered neighbours" of the
    revealed numbers around it; cells with no revealed neighbour get the global density.
    It has no global constraint reasoning, cannot prove a cell safe, and never combines
    components - which is exactly the gap the exact solver closes.
    """

    name = "min_prob"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, board: Board) -> Move:
        rows, cols = board.rows, board.cols
        cells = board.cells
        openable = board.covered()
        if not openable.any():
            return Move("done", (), 0.0, "no covered cells")

        # global density fallback
        covered_n = int(openable.sum()) + int((cells == FLAG).sum())
        density = board.mines_left / covered_n if covered_n else 1.0

        score = np.full((rows, cols), np.nan)
        for r in range(rows):
            for c in range(cols):
                if not openable[r, c]:
                    continue
                local = []
                for rr, cc in _nbrs(rows, cols, r, c):
                    v = int(cells[rr, cc])
                    if v < 0:
                        continue
                    unk, flags = [], 0
                    for r2, c2 in _nbrs(rows, cols, rr, cc):
                        x = int(cells[r2, c2])
                        if x == FLAG:
                            flags += 1
                        elif x in (COVERED, QUESTION):
                            unk.append((r2, c2))
                    if unk:
                        local.append(max(0.0, (v - flags) / len(unk)))
                score[r, c] = float(np.mean(local)) if local else density

        flat = np.where(openable)
        vals = score[flat]
        bad = ~np.isfinite(vals)
        if bad.any():  # cells with no number around them fall back to the global density
            vals = np.where(bad, density, vals)
        best = int(np.argmin(vals))
        rc = (int(flat[0][best]), int(flat[1][best]))
        return Move("guess", (rc,), float(vals[best]), f"min_prob: local ratio {vals[best]:.3f}")


class LearnedGuessPolicy:
    """The exact solver, plus a learned scorer that only ranks *guesses*.

    Contract:
      * zero-risk moves come from the solver, untouched - the model is never consulted;
      * when a guess is unavoidable, the model re-ranks the solver's candidates and the
        lowest predicted mine probability wins;
      * any failure (no checkpoint, torch missing, bad output) falls back to the solver's
        own choice, so a broken model can never make the agent play worse.

    Measured honestly in `reports/learned_vs_exact_400.json`: a scorer trained by imitating
    the exact probabilities does NOT beat the exact solver at guessing, and on this harness a
    wrong re-rank shows up immediately as logic losses. That is why the fallback is the
    default and the learned path stays opt-in behind a checkpoint.
    """

    name = "learned"

    def __init__(self, seed: int = 0, model=None, weights: str | None = None):
        self.solver = Solver(seed=seed)
        self.model = model
        if self.model is None and weights:
            try:
                from .model import load_scorer

                self.model = load_scorer(weights)
            except Exception:  # noqa: BLE001 - a bad checkpoint must not break playing
                self.model = None

    def act(self, board: Board) -> Move:
        move = self.solver.decide(board)
        if self.model is None or move.kind != "guess":
            return move
        try:
            from .dataset import encode_window

            rows, cols = board.rows, board.cols
            covered = board.covered()
            # only cells touching a revealed number can be opened at minimal risk; the
            # rest are equivalent by symmetry, so ranking them is pointless
            candidates = []
            for r in range(rows):
                for c in range(cols):
                    if not covered[r, c]:
                        continue
                    if any(0 <= rr < rows and 0 <= cc < cols and int(board.cells[rr, cc]) >= 0
                           for rr, cc in _nbrs(rows, cols, r, c)):
                        candidates.append((r, c))
            if len(candidates) < 2:
                return move
            windows = [encode_window(board, r, c) for (r, c) in candidates]
            scores = self.model(windows)
            best = candidates[int(np.argmin(scores))]
            return Move("guess", (best,), move.p_mine, move.reason + " | ranked by model")
        except Exception:  # noqa: BLE001 - never let the model break a game
            return move
