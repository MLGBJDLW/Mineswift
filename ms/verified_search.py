"""Model-proposed guesses reviewed against exact terminal scores in small endgames."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch

from .belief_rl import batch_inputs
from .belief_search import RolloutActor
from .endgame import _observations, _Search, _worlds
from .solver import BudgetExceeded, Move


@dataclass(frozen=True)
class Limits:
    worlds: int = 512
    cells: int = 40
    enumeration: int = 24_000
    tree: int = 150_000


def compare_choices(board, hidden, worlds, incumbent, order, work):
    """Every returned score is exact for optimal subsequent play in these worlds."""
    search = _Search(worlds, _observations(board, hidden, worlds), board.mines_left, len(hidden), work)
    remaining = (1 << len(hidden)) - 1
    feasible = (1 << len(worlds)) - 1
    safe = [feasible & ~mines for mines in search.mine_worlds]
    try:
        original_wins = search._click(remaining, safe[incumbent], incumbent)
    except BudgetExceeded:
        return None
    best, best_wins = incumbent, original_wins
    complete = True
    for cell in order:
        if cell == incumbent or safe[cell].bit_count() <= best_wins:
            continue
        try:
            wins = search._click(remaining, safe[cell], cell)
        except BudgetExceeded:
            complete = False
            break
        if wins > best_wins:
            best, best_wins = cell, wins
    return best, original_wins, best_wins, complete


class VerifiedActor(RolloutActor):
    name = "belief_verified"

    def __init__(self, actor, seed):
        super().__init__(actor, seed)
        self.checked = self.complete = self.eligible = 0
        self.changes = []
        self.extra_seconds = 0.

    def position(self, board):
        self.current = super().position(board)
        return self.current

    @torch.no_grad()
    def act(self, board):
        move = super().act(board)
        if move.kind != "guess":
            return move
        count = self.solver.last_stats["solutions"]
        limits = Limits()
        if not 2 < count <= limits.worlds:
            return move
        hidden = tuple(map(int, np.flatnonzero(board.covered())))
        if len(hidden) > limits.cells:
            return move
        start = perf_counter()
        self.eligible += 1
        try:
            worlds = _worlds(board, hidden, count, limits.enumeration)
            r, c = move.cells[0]
            incumbent = hidden.index(r*board.cols + c)
            scores = self.actor(*batch_inputs([self.current]))[0].numpy()
            order = sorted(range(len(hidden)), key=lambda i: scores[hidden[i]], reverse=True)
            result = compare_choices(board, hidden, worlds, incumbent, order, limits.tree)
            if result is None:
                return move
            cell, original_wins, wins, complete = result
            self.checked += 1
            self.complete += int(complete)
            self.last_guess.update(verified_worlds=count, optimal_wins=wins)
            if cell == incumbent:
                return move
            chosen = hidden[cell]
            risk = float(self.current.planes[0].flat[chosen])
            self.last_guess.update(mine_risk=risk, rollout_worlds=0, rollout_wins=None)
            self.changes.append(dict(guess=self.guesses, worlds=count, incumbent_wins=original_wins,
                                     new_wins=wins, incumbent_cell=hidden[incumbent], new_cell=chosen,
                                     incumbent_risk=move.p_mine, new_risk=risk, complete=complete))
            return Move("guess", (divmod(chosen, board.cols),), risk, "model candidate with higher exact terminal value")
        except BudgetExceeded:
            return move
        finally:
            self.extra_seconds += perf_counter() - start
