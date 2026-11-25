"""From-scratch terminal-reward guess policy. Run python -m ms.rl {train,compare} --help."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from .board import Board
from .engine import EXPERT, Config
from .eval import mcnemar, play_game, run, summarize
from .opening import first_cell
from .solver import Move, Solver


class BoardPolicyNet(nn.Module):
    """One logit per cell, conditioned on the visible grid and remaining mine counter."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(13, 8)  # Board codes -4..8
        self.local = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.context = nn.Linear(17, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, board: Board) -> torch.Tensor:
        tokens = torch.from_numpy(board.cells).long() + 4
        covered = torch.from_numpy(board.covered())
        x = self.embedding(tokens).permute(2, 0, 1).unsqueeze(0)
        local = self.local(x)
        remaining = local.new_tensor([[board.mines_left / max(1, int(covered.sum()))]])
        summary = torch.cat((local.mean(dim=(2, 3)), remaining), dim=1)
        context = self.context(summary)[:, :, None, None]
        logits = self.head(torch.tanh(local + context)).reshape(-1)
        return logits.masked_fill(~covered.flatten(), -torch.inf)


class RLGuessPolicy:
    name = "rl"

    def __init__(self, seed: int, model: BoardPolicyNet, generator: torch.Generator | None = None):
        # Offline training can finish the proof before authorizing a guess.
        self.solver = Solver(seed=seed, max_nodes=None)
        self.model = model
        self.generator = generator  # supplied only during on-policy training
        self.log_probs: list[torch.Tensor] = []
        self.guesses = 0

    def act(self, board: Board) -> Move:
        move = self.solver.deduce(board)
        if move is not None:
            return move
        with torch.set_grad_enabled(self.generator is not None):
            logits = self.model(board)
            if self.generator is None:
                cell = int(logits.argmax())
            else:
                log_probs = torch.log_softmax(logits, dim=0)
                cell = int(torch.multinomial(log_probs.detach().exp(), 1, generator=self.generator))
                self.log_probs.append(log_probs[cell])
        self.guesses += 1
        return Move("guess", (divmod(cell, board.cols),), None, "terminal-reward policy")


def episode_loss(log_probs: list[torch.Tensor], won: bool) -> torch.Tensor:
    """REINFORCE, with the same undiscounted terminal return for every sampled guess."""
    reward = 1 if won else -1
    return -reward * torch.stack(log_probs).sum()


def train(cfg: Config, games: int, seed_base: int, model_seed: int, batch: int, lr: float):
    if games < 1 or batch < 1 or min(seed_base, model_seed) < 0 or not (math.isfinite(lr) and lr > 0):
        raise ValueError("need positive games, batch and lr, and nonnegative seeds")
    torch.manual_seed(model_seed)
    model = BoardPolicyNet()
    generator = torch.Generator().manual_seed(model_seed + 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    initial = [p.detach().clone() for p in model.parameters()]
    batches = []
    wins = guesses = guess_games = guess_wins = degraded = updates = 0
    for start in range(0, games, batch):
        size = min(batch, games - start)
        optimizer.zero_grad(set_to_none=True)
        batch_wins = batch_guesses = 0
        for seed in range(seed_base + start, seed_base + start + size):
            policy = RLGuessPolicy(seed, model, generator)
            game = play_game(seed, policy, cfg, opening="random")
            if game.outcome not in ("win", "guess"):
                raise RuntimeError(f"unexpected training outcome {game.outcome} on seed {seed}")
            batch_wins += game.won
            degraded += game.degraded
            if policy.log_probs:
                (episode_loss(policy.log_probs, game.won) / size).backward()
                batch_guesses += policy.guesses
                guess_games += 1
                guess_wins += game.won
        if batch_guesses:
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True))
            optimizer.step()
            updates += 1
        else:
            grad_norm = 0.0
        batches.append(dict(games=size, wins=batch_wins, guesses=batch_guesses, grad_norm=grad_norm))
        wins += batch_wins
        guesses += batch_guesses
    with torch.no_grad():
        delta = sum(float((p - p0).square().sum()) for p, p0 in zip(model.parameters(), initial))
    report = dict(cfg=asdict(cfg), games=games, seed_base=seed_base, model_seed=model_seed,
                  action_seed=model_seed + 1, opening="random", batch=batch, lr=lr,
                  wins=wins, guesses=guesses, games_with_guesses=guess_games,
                  wins_with_guesses=guess_wins, proof_budget_exhaustions=degraded,
                  updates=updates, parameter_delta_l2=math.sqrt(delta), batches=batches)
    if not math.isfinite(delta) or delta == 0:
        raise RuntimeError("no finite parameter change from sampled guesses; try a larger training run")
    return model, report


def compare(checkpoint: Path, games: int, seed_base: int, jobs: int = 1) -> dict:
    if min(games, jobs) < 1 or seed_base < 0:
        raise ValueError("need positive games and jobs, and a nonnegative seed base")
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    training = saved["training"]
    train_base = training["seed_base"]
    if max(seed_base, train_base) < min(seed_base + games, train_base + training["games"]):
        raise ValueError("evaluation seeds overlap the training boards in the checkpoint")
    cfg = Config(**training["cfg"])
    model = BoardPolicyNet()
    model.load_state_dict(saved["model"])
    model.eval()
    baseline = run("endgame", games, cfg, seed_base, jobs, opening="random")
    results = []
    pairs = []
    start = perf_counter()
    for ref in baseline.results:
        policy = RLGuessPolicy(ref.seed, model)
        game = play_game(ref.seed, policy, cfg, opening="random")
        results.append(game)
        pairs.append(dict(seed=ref.seed, first_cell=first_cell(cfg, ref.seed, "random"),
                          endgame=ref.outcome, rl=game.outcome, rl_guesses=policy.guesses))
    challenger = summarize("rl", cfg, results, perf_counter() - start)
    return dict(cfg=asdict(cfg), opening="random", games=games, seed_base=seed_base,
                train_seed_base=train_base, train_games=training["games"], model_seed=training["model_seed"],
                checkpoint=str(checkpoint), guess_selection="greedy", policies={
                    "endgame": baseline.summary(), "rl": challenger.summary()},
                paired=mcnemar(baseline, challenger), boards=pairs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train_cmd = commands.add_parser("train")
    train_cmd.add_argument("--games", type=int, default=512)
    train_cmd.add_argument("--seed-base", type=int, default=470000)
    train_cmd.add_argument("--model-seed", type=int, default=17)
    train_cmd.add_argument("--batch", type=int, default=16)
    train_cmd.add_argument("--lr", type=float, default=0.003)
    compare_cmd = commands.add_parser("compare")
    compare_cmd.add_argument("--games", type=int, default=256)
    compare_cmd.add_argument("--seed-base", type=int, default=480000)
    compare_cmd.add_argument("--jobs", type=int, default=1, help="workers for endgame baseline")
    for cmd in (train_cmd, compare_cmd):
        cmd.add_argument("--checkpoint", type=Path, default=Path("models/rl_guess.pt"))
        cmd.add_argument("--out", type=Path, help="write full JSON report")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.command == "train":
        model, report = train(EXPERT, args.games, args.seed_base, args.model_seed, args.batch, args.lr)
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "training": report}, args.checkpoint)
    else:
        report = compare(args.checkpoint, args.games, args.seed_base, args.jobs)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("boards", "batches")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
