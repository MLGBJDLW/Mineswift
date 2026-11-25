"""Learn differences in terminal win probability from paired public-belief continuations."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import lru_cache, partial
from hashlib import sha256
import json
from pathlib import Path
from random import Random

import numpy as np
import torch

from experiments.belief_rl import init_worker
from ms.verified_search import Limits, VerifiedActor
from ms.belief_rl import BeliefActor, BeliefPolicy, _neighbors, batch_inputs, next_guess
from ms.belief_search import RolloutActor
from ms.engine import EXPERT, Minefield
from ms.eval import GameResult, mcnemar, play_game, summarize
from ms.opening import first_cell
from ms.play_web import WEIGHTS
from ms.policy import EndgamePolicy
from ms.posterior import sample_layouts
from ms.rollout import _finish, _seed
from ms.solver import Move


FEATURES = ("survival", "actor_margin", "safe_zero", "safe_digit_entropy",
            "safe_decides_neighbors", "covered_neighbors", "edge_neighbors")
MIN_EDGE = .025
LIMIT = 120


@lru_cache(maxsize=1)
def frozen_actor():
    net = BeliefActor()
    net.load_state_dict(torch.load(WEIGHTS, weights_only=True, map_location="cpu")["model"])
    return net.eval()


def eligible(size, worlds):
    return size <= LIMIT and (size > Limits().cells or worlds > Limits().worlds)


@torch.no_grad()
def proposals(board, state, incumbent, actor):
    """Form a small diverse slate from the actor and public conditional observations."""
    layouts = sample_layouts(board, 64, Random(_seed(board)))
    if layouts is None:
        return None
    cells = np.flatnonzero(board.covered())
    risk = state.planes[0].ravel()[cells]
    nb = _neighbors(board.rows, board.cols)[cells]
    hidden = np.append(board.covered(), False)[nb].sum(axis=1)
    flags = np.append(board.flags(), False)[nb].sum(axis=1)
    mines = np.stack(layouts)
    digits = np.pad(mines, ((0, 0), (0, 1)))[:, nb].sum(axis=2)
    safe = ~mines[:, cells]
    nsafe = np.maximum(1, safe.sum(axis=0))
    hist = np.stack([((digits == d) & safe).sum(axis=0) / nsafe for d in range(9)])
    entropy = -(hist * np.log2(np.maximum(hist, 1e-30))).sum(axis=0)
    decides = (hidden > 0) & ((digits == flags) | (digits == flags + hidden))
    progress = (safe & decides).sum(axis=0) / nsafe
    logits = actor(*batch_inputs([state]))[0].numpy()[cells]
    logits = np.clip(logits - logits.max(), -12., 0.)
    features = np.column_stack((1-risk, logits, hist[0], entropy, progress, hidden/8, (nb < 0).sum(1)/8))
    near = risk <= risk.min() + .06
    first = int(np.searchsorted(cells, incumbent))
    # Keep row-major tie breaking deterministic, including on equal posterior risk.
    slate = [first]
    for priority in (-risk, progress - 2*risk, entropy - 3*risk, hist[0] - 2*risk):
        idx = int(np.argmax(np.where(near, priority, -np.inf)))
        if idx not in slate:
            slate.append(idx)
    return cells[slate], features[slate]


def training_row(seed, draws):
    actor = frozen_actor()
    policy = BeliefPolicy(actor, seed)
    field = Minefield(EXPERT, np.random.default_rng(seed*1_000_003 + 7))
    field.click(first_cell(EXPERT, seed, "random"))
    skip = seed % 2
    while not field.won:
        state = next_guess(field, policy)
        if state is None:
            return None
        first = policy.choose(state)
        board = field.board()
        worlds = policy.solver.last_stats["solutions"]
        if eligible(int(board.covered().sum()), worlds):
            if skip == 0:
                proposal = proposals(board, state, first, actor)
                if proposal is not None and len(proposal[0]) > 1:
                    break
            else:
                skip -= 1
        if field.click(first) == "boom":
            return None
    else:
        return None
    cells, features = proposal
    # Labels and observation features use independent sampling streams.
    layouts = sample_layouts(board, draws, Random(_seed(board) ^ 0x916BCD1A72))
    if layouts is None:
        raise RuntimeError("sampling feasibility changed on the same board")
    simulation = Minefield(EXPERT, np.random.default_rng(0))
    follow_policy = BeliefPolicy(actor, seed)
    moves = {}

    def follow(public):
        key = public.cells.tobytes()
        if key not in moves:
            moves[key] = follow_policy.act(public)
        return moves[key]

    outcomes = []
    for layout in layouts:
        simulation.load(layout)
        trial = []
        for cell in cells:
            simulation.visible[:] = board.cells.ravel()
            simulation.won = simulation.exploded = False
            won = _finish(simulation, int(cell), follow)
            if won is None:
                raise RuntimeError(f"unfinished training branch on {seed}")
            trial.append(int(won))
        outcomes.append(trial)
    return dict(seed=seed, cells=cells.tolist(), features=features.tolist(), outcomes=outcomes)


def fit(rows):
    """Equal weight per position, fit differences against the incumbent using final wins."""
    inputs, targets, weights = [], [], []
    for row in rows:
        x = np.asarray(row["features"])
        y = np.mean(row["outcomes"], axis=0)
        for j in range(1, len(y)):
            inputs.append(x[j]-x[0])
            targets.append(y[j]-y[0])
            weights.append(1/(len(y)-1))
    x, y, w = np.array(inputs), np.array(targets), np.array(weights)
    if not len(x):
        raise ValueError("no eligible paired positions")
    scale = np.maximum(np.sqrt(np.average(x*x, weights=w, axis=0)), .01)
    z = x / scale
    gram = z.T @ (w[:, None] * z)
    return np.linalg.solve(gram + .1*w.sum()*np.eye(x.shape[1]), z.T @ (w*y)) / scale


class MidgameActor(VerifiedActor):
    def __init__(self, actor, seed, coefficients):
        super().__init__(actor, seed)
        self.coefficients = coefficients
        self.mid_changes = self.mid_checked = 0

    def act(self, board):
        move = super().act(board)
        if move.kind != "guess" or not eligible(int(board.covered().sum()), self.solver.last_stats["solutions"]):
            return move
        first = int(np.ravel_multi_index(move.cells[0], board.cells.shape))
        proposal = proposals(board, self.current, first, self.actor)
        if proposal is None:
            return move
        cells, features = proposal
        self.mid_checked += 1
        values = (features - features[0]) @ self.coefficients
        j = int(np.argmax(values))
        if values[j] < MIN_EDGE:
            return move
        self.mid_changes += 1
        cell = int(cells[j])
        risk = float(self.current.planes[0].flat[cell])
        self.last_guess.update(mine_risk=risk, midgame_estimated_gain=float(values[j]))
        return Move("guess", (divmod(cell, board.cols),), risk, "paired terminal-return model")


def evaluate(seed, coefficients):
    net = frozen_actor()
    def run(policy):
        result = play_game(seed, policy, EXPERT, opening="random")
        if result.outcome not in ("win", "guess"):
            raise RuntimeError(f"invalid outcome {result.outcome} on {seed}")
        return result.won

    new = MidgameActor(net, seed, coefficients)
    challenger = run(new)
    # Until the first changed action these deterministic policies have identical trajectories.
    verified = VerifiedActor(net, seed) if new.mid_changes else new
    previous = run(verified) if new.mid_changes else challenger
    live = run(RolloutActor(net, seed)) if verified.changes else previous
    return dict(seed=seed, mid=challenger, verified=previous, live=live, endgame=run(EndgamePolicy(seed)),
                checked=new.mid_checked, changed=new.mid_changes)


def summary(rows):
    stats = {}
    for name in ("mid", "verified", "live", "endgame"):
        results = [GameResult(r["seed"], r[name], 0, "win" if r[name] else "guess") for r in rows]
        stats[name] = summarize(name, EXPERT, results, 0)
    return dict(wins={k:v.wins for k,v in stats.items()}, paired=[
        mcnemar(stats[k], stats["mid"]) for k in ("verified", "live", "endgame")],
        verified_vs_live=mcnemar(stats["live"], stats["verified"]),
        changed_boards=sum(bool(r["changed"]) for r in rows), changed=sum(r["changed"] for r in rows),
        checked=sum(r["checked"] for r in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("train", "eval"))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--games", type=int, required=True)
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--draws", type=int, default=16)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if min(args.jobs, args.games, args.draws) < 1 or args.seed < 0:
        p.error("games, jobs and draws must be positive and seeds nonnegative")
    parent = torch.load(WEIGHTS, weights_only=True, map_location="cpu")["training"]
    if max(args.seed, parent["seed_base"]) < min(args.seed+args.games, parent["seed_base"]+parent["layouts"]):
        p.error("requested boards overlap frozen actor training")
    fingerprint = sha256(WEIGHTS.read_bytes()).hexdigest()
    if args.stage == "train":
        work = partial(training_row, draws=args.draws)
    else:
        model = json.loads(args.model.read_text())
        if model["actor_sha256"] != fingerprint or model["features"] != list(FEATURES):
            p.error("incompatible model")
        if model["min_edge"] != MIN_EDGE or model["cell_limit"] != LIMIT:
            p.error("incompatible switch threshold or board size limit")
        if max(args.seed, model["seed"]) < min(args.seed+args.games, model["seed"]+model["games"]):
            p.error("evaluation overlaps training seeds")
        work = partial(evaluate, coefficients=np.array(model["coefficients"]))
    init_worker()
    rows = []
    with ProcessPoolExecutor(args.jobs, initializer=init_worker) as pool:
        for i, row in enumerate(pool.map(work, range(args.seed, args.seed+args.games), chunksize=4), 1):
            if row is not None:
                rows.append(row)
            if i % 256 == 0:
                print(i, flush=True)
    metadata = dict(seed=args.seed, games=args.games, actor_sha256=fingerprint)
    if args.stage == "train":
        model = dict(**metadata, draws=args.draws, positions=len(rows), features=FEATURES,
                     coefficients=fit(rows).tolist(), min_edge=MIN_EDGE, cell_limit=LIMIT,
                     branches=sum(len(r["cells"])*len(r["outcomes"]) for r in rows))
        args.model.parent.mkdir(parents=True, exist_ok=True)
        args.model.write_text(json.dumps(model, indent=2)+"\n")
        print(model)
    else:
        metadata.update(cfg=asdict(EXPERT), opening="random",
                        model_sha256=sha256(args.model.read_bytes()).hexdigest(), **summary(rows))
        print(metadata)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(**metadata, rows=rows), separators=(",", ":"))+"\n")


if __name__ == "__main__":
    main()
