"""Terminal-reward actor with visible-board belief features and grouped policy gradients."""
from __future__ import annotations

from collections import Counter
from copy import copy
from dataclasses import dataclass
from functools import lru_cache
from math import log2
from random import Random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .board import Board
from .endgame import _observations, _worlds
from .engine import Config, Minefield, neighbour_ids
from .opening import first_cell
from .posterior import sample_layouts
from .rollout import _seed
from .solver import BudgetExceeded, Cell, Move, Solver


@dataclass
class Observation:
    cells: np.ndarray
    planes: np.ndarray


@lru_cache(maxsize=8)
def _neighbors(rows, cols):
    return neighbour_ids(rows, cols)  # -1 indexes an appended off-board zero


def sampled_reveals(board, planes):
    layouts = sample_layouts(board, 32, Random(_seed(board)))
    if layouts is None:
        return
    mines = np.stack(layouts)
    safe = ~mines
    nb = _neighbors(board.rows, board.cols)
    digits = np.pad(mines, ((0, 0), (0, 1)))[:, nb].sum(axis=2)
    legal = board.covered()
    flags = np.append(board.flags(), False)[nb].sum(axis=1)
    covered = np.append(legal, False)[nb].sum(axis=1)
    n = np.maximum(1, safe.sum(axis=0))
    hist = np.stack([((digits == k) & safe).sum(axis=0) / n for k in range(9)])
    entropy = -(hist * np.log2(np.maximum(hist, 1e-30))).sum(axis=0)
    decides = (covered > 0) & ((digits == flags) | (digits == flags + covered))
    valid = legal.ravel() & safe.any(axis=0)
    planes[2].ravel()[valid] = -1  # sampled safe-digit distribution
    planes[3].ravel()[valid] = entropy[valid]
    planes[4].ravel()[valid] = ((decides & safe).sum(axis=0) / n)[valid]
    planes[5].ravel()[valid] = hist[0, valid]


def observe(board: Board, belief: dict[Cell, float] | None, count: int, sample_large=False) -> Observation:
    """Encode visible clues, exact marginal risk and possible safe-click observations."""
    legal = board.covered()
    planes = np.zeros((6, board.rows, board.cols), dtype=np.float32)
    planes[1].fill(board.mines_left / int(legal.sum()))
    planes[0][board.flags()] = 1
    if belief is None:
        planes[0][legal] = planes[1][legal]  # public counter only, identical for all legal cells
        return Observation(board.cells, planes)
    for rc, risk in belief.items():
        planes[0][rc] = risk
    hidden = tuple(map(int, np.flatnonzero(legal)))
    if count <= 96 and len(hidden) <= 48:
        try:
            worlds = _worlds(board, hidden, count, 6000)
        except BudgetExceeded:
            if sample_large:
                sampled_reveals(board, planes)
            return Observation(board.cells, planes)
        reveals = _observations(board, hidden, worlds)
        planes[2][legal] = 1  # exact observation distribution available
        for i, cell in enumerate(hidden):
            groups = Counter(row[i] for row, mines in zip(reveals, worlds) if not (mines >> i) & 1)
            safe = sum(groups.values())
            # Entropy, chance of identifying the layout, mean safe flood size.
            planes[3].flat[cell] = log2(safe) - sum(n * log2(n) for n in groups.values()) / safe
            planes[4].flat[cell] = sum(n == 1 for n in groups.values()) / safe
            planes[5].flat[cell] = log2(sum(key[0].bit_count() * n for key, n in groups.items()) / safe)
    elif sample_large:
        sampled_reveals(board, planes)
    return Observation(board.cells, planes)


class BeliefActor(nn.Module):
    """Reflection-equivariant full-board actor. All action preferences are trainable."""

    def __init__(self):
        super().__init__()
        self.encode = nn.Conv2d(42, 24, 1)
        self.context = nn.Linear(24, 24)
        self.readout = nn.Conv2d(24, 1, 1)
        self.skip = nn.Conv2d(42, 1, 1, bias=False)
        # Start with a uniform distribution over all legal actions.
        nn.init.zeros_(self.skip.weight)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, cells: torch.Tensor, planes: torch.Tensor):
        tokens = F.one_hot(cells.long() + 4, 13).permute(0, 3, 1, 2).float()
        risk = planes[:, :1].clamp(1e-6, 1 - 1e-6)
        visible = torch.cat((tokens, planes, risk.log(), (-risk).log1p()), dim=1)
        neighborhood = F.avg_pool2d(visible, 3, stride=1, padding=1)
        features = torch.cat((visible, neighborhood), dim=1)
        local = torch.tanh(self.encode(features))
        context = self.context(local.mean(dim=(2, 3)))[:, :, None, None]
        scores = self.skip(features) + self.readout(torch.tanh(local + context))
        return scores.flatten(1).masked_fill(~((cells == -1) | (cells == -3)).flatten(1), -torch.inf)

    def action_log_probs(self, scores):
        return scores.log_softmax(0)


class ReturnActor(BeliefActor):
    """Log P(win) = log P(safe) + learned log P(win | safe, board, action)."""

    def forward(self, cells, planes):
        conditional = super().forward(cells, planes)
        survival = (1 - planes[:, 0].flatten(1)).clamp_min(1e-12)
        return survival.log() + F.logsigmoid(conditional)

    def action_log_probs(self, scores):
        # Explore every legal action, while exploiting estimated terminal win probabilities.
        legal = scores.isfinite()
        probs = 0.95 * (scores / 0.12).softmax(0) + 0.05 * legal / legal.sum()
        return probs.log()


def batch_inputs(observations):
    return (torch.from_numpy(np.stack([o.cells for o in observations])),
            torch.from_numpy(np.stack([o.planes for o in observations])))


@dataclass
class Decision:
    obs: Observation
    cell: int
    old_logp: float


class BeliefPolicy:
    name = "belief_rl"

    def __init__(self, actor: BeliefActor, seed: int, raw_only: bool = False):
        self.actor = actor
        self.solver = Solver(seed=seed, max_nodes=None)
        self.guesses = 0
        self.raw_only = raw_only

    def position(self, board):
        belief = None if self.raw_only else {}
        move = self.solver.deduce(board, belief=belief)
        if move is not None:
            return move
        return observe(board, belief, self.solver.last_stats["solutions"], isinstance(self.actor, ReturnActor))

    @torch.no_grad()
    def choose(self, obs, rng=None):
        logits = self.actor(*batch_inputs([obs]))[0]
        self.guesses += 1
        if rng is None:
            return int(logits.argmax())
        logp = self.actor.action_log_probs(logits)
        cell = int(torch.multinomial(logp.exp(), 1, generator=rng))
        return Decision(obs, cell, float(logp[cell]))

    def act(self, board):
        state = self.position(board)
        if isinstance(state, Move):
            return state
        cell = self.choose(state)
        return Move("guess", (divmod(cell, board.cols),), None, "terminal-return belief actor")


def next_guess(field, policy):
    """Close all proven moves. A flood may open some cells from the same proof."""
    while not field.won:
        state = policy.position(field.board())
        if isinstance(state, Observation):
            return state
        if state.kind not in ("flag", "open"):
            raise RuntimeError(f"unexpected solver move {state.kind}")
        for r, c in state.cells:
            i = r * field.cfg.cols + c
            if state.kind == "flag":
                field.flag(i)
            elif field.click(i) == "boom":
                raise RuntimeError("solver opened a mine on a proven move")
    return None


def group_rollouts(cfg: Config, seed: int, actor, group: int, action_seed: int, raw_only: bool = False):
    start = Minefield(cfg, np.random.default_rng(seed * 1_000_003 + 7))
    start.click(first_cell(cfg, seed, "random"))
    policy = BeliefPolicy(actor, seed, raw_only=raw_only)
    initial = next_guess(start, policy)
    trajectories = []
    for trajectory in range(group):
        # Layout and adjacency are read-only after placement. Clone the mutable observation.
        field = copy(start)
        field.visible = start.visible.copy()
        stream = np.random.SeedSequence([action_seed, seed, trajectory])
        rng = torch.Generator().manual_seed(int(stream.generate_state(1, dtype=np.uint64)[0]))
        decisions = []
        obs = initial
        while obs is not None:
            decision = policy.choose(obs, rng)
            decisions.append(decision)
            if field.click(decision.cell) == "boom":
                break
            obs = next_guess(field, policy)
        if not (field.won or field.exploded):
            raise RuntimeError(f"nonterminal rollout on {seed}")
        trajectories.append((1 if field.won else -1, decisions))
    return trajectories


def loo_advantages(rewards):
    rewards = np.asarray(rewards, dtype=float)
    if len(rewards) < 2:
        raise ValueError("leave-one-out needs at least two independent rollouts")
    return (len(rewards) * rewards - rewards.sum()) / (len(rewards) - 1)
