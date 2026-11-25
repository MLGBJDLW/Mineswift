"""One-step policy improvement by playing a frozen actor to terminal on small public beliefs."""
from __future__ import annotations

import numpy as np
import torch

from .belief_rl import BeliefPolicy, batch_inputs
from .endgame import _worlds
from .engine import Config, Minefield
from .rollout import _finish
from .solver import BudgetExceeded, Move


class RolloutActor(BeliefPolicy):
    name = "belief_rollout"

    def __init__(self, actor, seed, *, world_limit=32, cell_limit=24, candidate_limit=8):
        super().__init__(actor, seed)
        self.continuation = BeliefPolicy(actor, seed)
        self.searches = self.switches = 0
        self.world_limit, self.cell_limit, self.candidate_limit = world_limit, cell_limit, candidate_limit
        self.last_guess = {}

    @torch.no_grad()
    def act(self, board):
        state = self.position(board)
        if isinstance(state, Move):
            return state
        logits = self.actor(*batch_inputs([state]))[0]
        base = int(logits.argmax())
        self.guesses += 1
        pick = base
        count = self.solver.last_stats["solutions"]
        hidden = tuple(map(int, np.flatnonzero(board.covered())))
        searched = 0
        rollout_wins = None
        if 2 < count <= self.world_limit and len(hidden) <= self.cell_limit:
            # Sort only legal cells. Equal scores retain the actor's row-major tie rule.
            scores = logits.numpy()
            candidates = sorted(hidden, key=lambda cell: scores[cell], reverse=True)[:self.candidate_limit]
            try:
                worlds = _worlds(board, hidden, count, 6000)
            except BudgetExceeded:
                worlds = []
            if worlds:
                wins = self._terminal_wins(board, hidden, worlds, candidates)
                pick = candidates[int(np.argmax(wins))]
                self.searches += 1
                self.switches += int(pick != base)
                searched = len(worlds)
                rollout_wins = max(wins)
        risk = float(state.planes[0].flat[pick])
        self.last_guess = dict(mine_risk=risk, min_risk=float(state.planes[0][board.covered()].min()),
                               rollout_worlds=searched, rollout_wins=rollout_wins)
        reason = "belief actor with terminal rollouts" if searched else "belief actor"
        return Move("guess", (divmod(pick, board.cols),), risk, reason)

    def _terminal_wins(self, board, hidden, worlds, candidates):
        flags = board.flags().ravel()
        cfg = Config(board.rows, board.cols, int(flags.sum()) + board.mines_left, False)
        field = Minefield(cfg, np.random.default_rng(0))
        wins = [0] * len(candidates)
        moves = {}

        def follow(visible):
            key = visible.cells.tobytes()
            if key not in moves:
                moves[key] = self.continuation.act(visible)
            return moves[key]

        for mask in worlds:
            layout = flags.copy()
            for i, cell in enumerate(hidden):
                layout[cell] = bool(mask & (1 << i))
            field.load(layout)
            for j, first in enumerate(candidates):
                if layout[first]:
                    continue
                field.won = field.exploded = False
                field.visible[:] = board.cells.ravel()
                won = _finish(field, first, follow)
                if won is None:
                    raise RuntimeError("unfinished actor continuation on an enumerated layout")
                wins[j] += won
        return wins
