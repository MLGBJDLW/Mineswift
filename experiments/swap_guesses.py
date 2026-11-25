"""Review model guesses when swapping two cells is unobservable in some layouts."""
from functools import lru_cache
from random import Random

import numpy as np
import torch

from ms.belief_rl import _neighbors, batch_inputs
from ms.posterior import sample_layouts
from ms.rollout import _seed
from ms.solver import Move
from ms.verified_search import Limits, VerifiedActor


@lru_cache(maxsize=8)
def pair_geometry(rows, cols):
    nb = _neighbors(rows, cols)
    pairs, outside = [], []
    for a in range(rows*cols):
        neighbors = set(nb[a]) - {-1}
        for b in sorted(neighbors):
            if b <= a or (a//cols != b//cols and a%cols != b%cols):
                continue
            # Only these outside cells could distinguish a mine at a from one at b.
            difference = sorted((neighbors ^ (set(nb[b]) - {-1})) - {a, b})
            pairs.append((a, b))
            outside.append(difference + [-1]*(6-len(difference)))
    return np.asarray(pairs, dtype=int).reshape(-1, 2), np.asarray(outside, dtype=int).reshape(-1, 6)


def possible_swaps(board):
    pairs, outside = pair_geometry(board.rows, board.cols)
    covered = np.append(board.covered(), False)
    flags = np.append(board.flags(), True)  # padding counts as an already satisfied requirement
    unknown = covered[outside]
    valid = covered[pairs].all(1) & (flags[outside] | unknown).all(1) & (unknown.sum(1) <= 2)
    return pairs[valid], outside[valid]


def swap_credit(layouts, pairs, outside):
    mines = np.pad(np.stack(layouts), ((0, 0), (0, 1)), constant_values=True)
    ambiguous = (mines[:, pairs[:, 0]] != mines[:, pairs[:, 1]]) & mines[:, outside].all(axis=2)
    # Each ambiguous world has an equally likely swapped partner. Apply half
    # the Wilson lower bound and only the strongest witness for each cell.
    n = len(layouts)
    p = ambiguous.mean(axis=0)
    z = 1.96
    lower = (p + z*z/(2*n) - z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1+z*z/n)
    credit = np.zeros(mines.shape[1]-1)
    for side in pairs.T:
        np.maximum.at(credit, side, .5*np.maximum(0., lower))
    return credit


class SwapActor(VerifiedActor):
    name = "belief_conditional_swaps"

    def __init__(self, actor, seed):
        super().__init__(actor, seed)
        self.expanded = self.rechecked = self.review_switches = 0

    @torch.no_grad()
    def act(self, board):
        move = super().act(board)
        if move.kind != "guess":
            return move
        worlds = self.solver.last_stats["solutions"]
        if worlds <= 2 or (worlds <= Limits().worlds and board.covered().sum() <= Limits().cells):
            return move
        pairs, outside = possible_swaps(board)
        if not len(pairs):
            return move
        self.expanded += 1
        layouts = sample_layouts(board, 96, Random(_seed(board)))
        if layouts is None:
            return move
        self.rechecked += 1
        credit = swap_credit(layouts, pairs, outside)
        risk = self.current.planes[0].ravel()
        first = int(np.ravel_multi_index(move.cells[0], board.cells.shape))
        adjusted = risk - credit
        viable = (credit > 0) & (adjusted <= adjusted[first] - .04)
        if not viable.any():
            return move
        near = viable & (adjusted <= adjusted[viable].min() + .01)
        logits = self.actor(*batch_inputs([self.current]))[0].numpy()
        pick = int(np.argmax(np.where(near, logits, -np.inf)))
        self.review_switches += 1
        self.last_guess.update(mine_risk=float(risk[pick]), swap_credit=float(credit[pick]))
        return Move("guess", (divmod(pick, board.cols),), float(risk[pick]), "actor with conditional swap risk")
