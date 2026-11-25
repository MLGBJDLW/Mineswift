"""Train or compare the grouped terminal-reward actor on random-opening expert boards."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import torch

from ms.belief_rl import BeliefActor, BeliefPolicy, ReturnActor, batch_inputs, group_rollouts, loo_advantages
from ms.belief_search import RolloutActor
from ms.engine import EXPERT
from ms.eval import mcnemar, play_game, summarize
from ms.opening import first_cell
from ms.policy import EndgamePolicy
from ms.rl import BoardPolicyNet, RLGuessPolicy


def init_worker():
    torch.set_num_threads(1)


def pool_for(jobs):
    return get_context("spawn").Pool(jobs, initializer=init_worker) if jobs > 1 else nullcontext(None)


def cpu_weights(model):
    return {k: v.detach().numpy().copy() for k, v in model.state_dict().items()}


def restore(model, weights):
    model.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
    return model.eval()


def collect(work):
    weights, seed, group, action_seed, raw_only, value = work
    actor = restore(ReturnActor() if value else BeliefActor(), weights)
    trajectories = group_rollouts(EXPERT, seed, actor, group, action_seed, raw_only)
    rewards = [r for r, _ in trajectories]
    if value:
        samples = [(d, float(r == 1)) for r, decisions in trajectories for d in decisions]
    else:
        samples = [(d, float(a)) for (_, decisions), a in zip(trajectories, loo_advantages(rewards))
                   if a != 0 for d in decisions]
    return rewards.count(1), sum(len(ds) for _, ds in trajectories), samples


def optimize(actor, optimizer, samples, episodes, epochs, metrics=None):
    if not samples:
        return 0.0
    for _ in range(epochs):
        epoch_loss = kl_sum = clipped = 0.0
        optimizer.zero_grad(set_to_none=True)
        for i in range(0, len(samples), 64):
            chunk = samples[i:i + 64]
            decisions, adv = zip(*chunk)
            target = torch.tensor(adv, dtype=torch.float32)
            scores = actor(*batch_inputs([d.obs for d in decisions]))
            ix = (torch.arange(len(chunk)), [d.cell for d in decisions])
            if isinstance(actor, ReturnActor):
                log_win = scores[ix]
                log_loss = torch.log1p(-log_win.exp().clamp_max(1 - 1e-7))
                loss = -torch.where(target.bool(), log_win, log_loss).sum() / len(samples)
            else:
                old = torch.tensor([d.old_logp for d in decisions])
                ratio = (scores.log_softmax(1)[ix] - old).exp()
                objective = torch.minimum(ratio * target, ratio.clamp(0.8, 1.2) * target)
                loss = -objective.sum() / episodes
                if metrics is not None:
                    with torch.no_grad():
                        kl_sum += float((ratio - 1 - (scores.log_softmax(1)[ix] - old)).sum())
                        clipped += int(((ratio < .8) | (ratio > 1.2)).sum())
            if metrics is not None:
                epoch_loss += float(loss.detach())
            loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if metrics is not None:
            metrics.append(dict(loss=epoch_loss, grad_norm=float(norm), approx_kl=kl_sum/len(samples),
                                clipped_fraction=clipped/len(samples)) if not isinstance(actor, ReturnActor)
                           else dict(loss=epoch_loss, grad_norm=float(norm)))
    return float(norm)


def train(args):
    if args.group < 2 or min(args.layouts, args.batch, args.epochs) <= 0:
        raise ValueError("positive layout, batch and epoch counts and group >= 2 required")
    torch.manual_seed(args.model_seed)
    actor = ReturnActor() if args.value else BeliefActor()
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)
    initial = cpu_weights(actor)
    batches = []
    with pool_for(args.jobs) as pool:
        for start in range(0, args.layouts, args.batch):
            size = min(args.batch, args.layouts - start)
            weights = cpu_weights(actor)
            tasks = [(weights, args.seed_base + start + j, args.group, args.model_seed + 1, args.raw_only, args.value)
                     for j in range(size)]
            output = list(pool.map(collect, tasks) if pool else map(collect, tasks))
            samples = [item for _, _, data in output for item in data]
            diagnostics = []
            norm = optimize(actor, optimizer, samples, size * args.group, args.epochs, diagnostics)
            batches.append(dict(layouts=size, wins=sum(w for w, _, _ in output),
                                guesses=sum(g for _, g, _ in output), samples=len(samples),
                                mixed_groups=sum(0 < w < args.group for w, _, _ in output),
                                last_grad_norm=norm, optimizer_epochs=diagnostics))
            if len(batches) % 8 == 0:
                print(f"{start + size}/{args.layouts} layouts", flush=True)
    delta = sum(float((v - torch.from_numpy(initial[k])).square().sum())
                for k, v in actor.state_dict().items()) ** 0.5
    if not delta > 0:
        raise RuntimeError("no policy update from this run")
    report = dict(cfg=asdict(EXPERT), opening="random", seed_base=args.seed_base,
                  layouts=args.layouts, episodes=args.layouts * args.group, group=args.group,
                  model_seed=args.model_seed, batch=args.batch, epochs=args.epochs, lr=args.lr,
                  raw_only=args.raw_only, value=args.value, parameter_delta_l2=delta, batches=batches)
    for key in ("wins", "guesses", "mixed_groups", "samples"):
        report[key] = sum(b[key] for b in batches)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=actor.state_dict(), training=report), args.checkpoint)
    return report


def replay(work):
    seed, weights, old_weights, control, rollout, value_weights = work
    actor = BeliefPolicy(restore(BeliefActor(), weights), seed)
    old = RLGuessPolicy(seed, restore(BoardPolicyNet(), old_weights))
    policies = [EndgamePolicy(seed), old, actor]
    if control is not None:
        policies.append(BeliefPolicy(restore(BeliefActor(), control), seed, raw_only=True))
    search = RolloutActor(actor.actor, seed) if rollout else None
    if search is not None:
        policies.append(search)
    value_search = None
    if value_weights is not None:
        value_net = restore(ReturnActor(), value_weights)
        policies.append(BeliefPolicy(value_net, seed))
        if rollout:
            value_search = RolloutActor(value_net, seed)
            policies.append(value_search)
    results = [play_game(seed, p, EXPERT, opening="random") for p in policies]
    counts = {}
    for name, policy in (("belief", search), ("value", value_search)):
        if policy:
            counts.update({f"{name}_searches": policy.searches, f"{name}_switches": policy.switches})
    return results, counts


def compare(args):
    if args.games < 1:
        raise ValueError("positive test size required")
    saved = torch.load(args.checkpoint, weights_only=True)
    old = torch.load(args.old_checkpoint, weights_only=True)
    control = torch.load(args.raw_checkpoint, weights_only=True) if args.raw_checkpoint else None
    value = torch.load(args.value_checkpoint, weights_only=True) if args.value_checkpoint else None
    if saved["training"].get("value") or saved["training"].get("raw_only"):
        raise ValueError("use a belief policy-gradient model as the primary checkpoint")
    checkpoints = [(saved, "layouts"), (old, "games")]
    paths = [args.checkpoint, args.old_checkpoint]
    names = ["endgame", "rl", "belief_rl"]
    if control is not None:
        if not control["training"]["raw_only"]:
            raise ValueError("expected a belief actor and a separately trained raw control")
        checkpoints.append((control, "layouts"))
        paths.append(args.raw_checkpoint)
        names.append("raw_group_rl")
    if args.rollout:
        names.append("belief_rollout")
    if value is not None:
        if not value["training"]["value"]:
            raise ValueError("expected a terminal action-value checkpoint")
        checkpoints.append((value, "layouts"))
        paths.append(args.value_checkpoint)
        names.append("value_rl")
        if args.rollout:
            names.append("value_rollout")
    for checkpoint, n in checkpoints:
        metadata = checkpoint["training"]
        if metadata["cfg"] != asdict(EXPERT) or metadata["opening"] != "random":
            raise ValueError("the expert comparison requires expert random-opening checkpoints")
        base = metadata["seed_base"]
        if max(base, args.seed_base) < min(base + metadata[n], args.seed_base + args.games):
            raise ValueError("assessment and training seeds overlap")
    weights = {k: v.numpy() for k, v in saved["model"].items()}
    old_weights = {k: v.numpy() for k, v in old["model"].items()}
    raw_weights = {k: v.numpy() for k, v in control["model"].items()} if control is not None else None
    value_weights = {k: v.numpy() for k, v in value["model"].items()} if value is not None else None
    tasks = [(s, weights, old_weights, raw_weights, args.rollout, value_weights)
             for s in range(args.seed_base, args.seed_base + args.games)]
    start = perf_counter()
    output = []
    with pool_for(args.jobs) as pool:
        for row in (pool.imap(replay, tasks) if pool else map(replay, tasks)):
            output.append(row)
            if len(output) % 128 == 0:
                print(f"{len(output)}/{args.games} paired boards", flush=True)
    runs = [results for results, _ in output]
    policies = {n: summarize(n, EXPERT, results, 0) for n, results in zip(names, zip(*runs))}
    boards = []
    for games, counters in output:
        seed = games[0].seed
        if any(g.seed != seed or g.outcome not in ("win", "guess") for g in games):
            raise RuntimeError(f"invalid paired result on {seed}")
        boards.append(dict(seed=seed, first_cell=first_cell(EXPERT, seed, "random"),
                           **counters,
                           **{n: g.outcome for n, g in zip(names, games)}))
    summaries = {n: s.summary() for n, s in policies.items()}
    for s in summaries.values():
        del s["elapsed_s"]  # policies were interleaved across workers
    paired = [mcnemar(policies[n], policies["belief_rl"]) for n in ("endgame", "rl", "raw_group_rl")
              if n in policies]
    if args.rollout:
        paired.extend(mcnemar(policies[n], policies["belief_rollout"]) for n in ("endgame", "belief_rl", "rl"))
    if value is not None:
        paired.extend(mcnemar(policies[n], policies["value_rl"]) for n in ("endgame", "belief_rl", "rl"))
        if args.rollout:
            paired.extend(mcnemar(policies[n], policies["value_rollout"])
                          for n in ("endgame", "value_rl", "rl"))
    return dict(games=args.games, seed_base=args.seed_base, cfg=asdict(EXPERT), opening="random",
                checkpoints={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                elapsed_s=round(perf_counter() - start, 2), policies=summaries,
                paired=paired,
                boards=boards)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True, dest="command")
    t = sub.add_parser("train")
    t.add_argument("--layouts", type=int, default=2048)
    t.add_argument("--group", type=int, default=4)
    variant = t.add_mutually_exclusive_group()
    variant.add_argument("--raw-only", action="store_true", help="ablate all conditional layout features")
    variant.add_argument("--value", action="store_true", help="fit terminal action values with a survival factor")
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--lr", type=float, default=0.012)
    t.add_argument("--seed-base", type=int, default=500000)
    t.add_argument("--model-seed", type=int, default=29)
    e = sub.add_parser("compare")
    e.add_argument("--games", type=int, default=1024)
    e.add_argument("--seed-base", type=int, default=530000)
    e.add_argument("--old-checkpoint", type=Path, default=Path("models/rl_guess.pt"))
    e.add_argument("--raw-checkpoint", type=Path, help="include the matched-budget raw-board control")
    e.add_argument("--value-checkpoint", type=Path, help="include the terminal action-value actor")
    e.add_argument("--rollout", action="store_true", help="also evaluate the actor with small-board planning")
    for cmd in (t, e):
        cmd.add_argument("--jobs", type=int, default=8)
        cmd.add_argument("--checkpoint", type=Path)
        cmd.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.seed_base < 0:
        parser.error("jobs must be positive and seeds nonnegative")
    if args.checkpoint is None:
        name = "belief_rl"
        if args.command == "train" and args.value:
            name = "value_rl"
        elif args.command == "train" and args.raw_only:
            name = "raw_group_rl"
        args.checkpoint = Path("models") / (name + ".pt")
    init_worker()
    report = train(args) if args.command == "train" else compare(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("boards", "batches")}), flush=True)


if __name__ == "__main__":
    main()
