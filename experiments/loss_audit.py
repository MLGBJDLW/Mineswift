"""Post-mortem diagnostics. The live policy still sees only Board, never true mines."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ms.board import COVERED, FLAG, QUESTION
from ms.engine import EXPERT, Minefield
from ms.policy import EndgamePolicy


def swap_witness(mf: Minefield, dead: int, opening: int) -> int | None:
    """Find a safe partner whose swap is invisible on all other safe cells.

    This is stronger than two equal marginals: even revealing everything safe
    outside the pair cannot distinguish the two layouts. It does not imply
    that all marked losses were unavoidable, or give a global win-rate bound.
    """
    opening_zone = {opening, *(int(j) for j in mf.nb[opening] if j >= 0)}
    if dead in opening_zone or not mf.mines[dead]:
        return None
    if np.any((mf.visible == FLAG) & ~mf.mines):
        return None  # flags do not describe the actual world; no certificate
    neighbors = [frozenset(int(j) for j in row if j >= 0) for row in mf.nb]
    partners = np.flatnonzero(~mf.mines & np.isin(mf.visible, (COVERED, QUESTION)))
    for idx in partners:
        other = int(idx)
        if other in opening_zone:
            continue  # swapped layout must obey the same opening constraint
        distinguishing = (neighbors[dead] ^ neighbors[other]) - {dead, other}
        if all(mf.mines[j] for j in distinguishing):
            return other
    return None


def audit_game(seed: int) -> tuple[str, str, bool]:
    mf = Minefield(EXPERT, np.random.default_rng(seed * 1_000_003 + 7))
    opening = (EXPERT.rows // 2) * EXPERT.cols + EXPERT.cols // 2
    mf.click(opening)
    policy = EndgamePolicy(seed=seed)
    for _ in range(4000):
        if mf.won:
            return 'win', '', False
        board = mf.board()
        move = policy.act(board)
        if move.kind == 'done':
            return 'stuck', '', False
        if move.kind == 'flag':
            for r, c in move.cells:
                mf.flag(r * board.cols + c)
            continue
        r, c = move.cells[0]
        clicked = r * board.cols + c
        if mf.click(clicked) != 'boom':
            continue
        stats = policy.solver.last_stats
        if stats.get('degraded'):
            phase = 'sampled'
        elif move.kind != 'guess':
            phase = 'unexpected_certain_loss'
        elif stats.get('endgame_solved'):
            phase = 'exact_endgame'
        elif stats['solutions'] <= 160 and int(board.covered().sum()) <= 48:
            phase = 'inside_endgame_limits_but_not_solved'
        else:
            phase = 'outside_endgame_limits'
        return 'loss', phase, swap_witness(mf, clicked, opening) is not None
    return 'stuck', '', False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed-base', type=int, required=True)
    parser.add_argument('--games', type=int, default=1000)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    seeds = range(args.seed_base, args.seed_base + args.games)
    counts: Counter = Counter()
    with Pool(args.jobs) as pool:
        for outcome, phase, witness in pool.imap_unordered(audit_game, seeds, chunksize=8):
            counts[outcome, phase, witness] += 1
    report = {'cfg': asdict(EXPERT), 'seed_base': args.seed_base, 'games': args.games,
              'policy': 'endgame', 'jobs': args.jobs,
              'counts': [dict(outcome=outcome, phase=phase, swap_witness=witness, games=count)
                         for (outcome, phase, witness), count in sorted(counts.items())]}
    data = json.dumps(report, indent=2)
    with args.out.open('x', encoding='utf-8') as fh:
        fh.write(data + '\n')
    print(data, flush=True)


if __name__ == '__main__':
    main()
