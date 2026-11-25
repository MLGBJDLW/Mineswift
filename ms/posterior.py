"""Sample equally likely mine layouts conditional on a visible Minesweeper board."""
from __future__ import annotations

from collections import defaultdict
from math import comb
from random import Random

import numpy as np

from .board import COVERED, FLAG, QUESTION, Board
from .solver import BudgetExceeded, Inconsistent, components, enumerate_group, raw_constraints


def sample_layouts(board: Board, draws: int, rng: Random, *, max_nodes: int = 6000,
                   max_component_layouts: int = 3000) -> list[np.ndarray] | None:
    """Return iid flat boolean layouts, or None if exact sampling is not possible.

    Revealed digits, trusted flags, and the global mine count are the only evidence.
    Components are enumerated independently; we weight their per-count buckets by
    completion counts, including the binomial count of placements in free cells.
    Integer weights and randrange avoid loss of accuracy when counts are huge.
    """
    if draws < 1 or max_nodes < 0 or max_component_layouts < 1:
        raise ValueError('invalid sampling limits')
    if np.any((board.cells < QUESTION) | (board.cells > 8)):
        return None
    budget = int(board.mines_left)
    if not 0 <= budget <= int(board.covered().sum()):
        return None
    try:
        raw, _, free = raw_constraints(board, strict=True)
        # Strict mode also validates clues that no longer touch any unknowns.
        groups = []
        for component in components(raw):
            by_count: dict[int, list[int]] = defaultdict(list)
            count = 0

            def collect(layout: int) -> None:
                nonlocal count
                count += 1
                if count > max_component_layouts:
                    raise BudgetExceeded
                mines = layout.bit_count()
                if mines <= budget:
                    by_count[mines].append(layout)

            cells, _, _, _, _ = enumerate_group(component, max_nodes, on_solution=collect)
            groups.append((cells, sorted(by_count.items())))
    except (BudgetExceeded, Inconsistent, RecursionError):
        return None

    free_cells = sorted(free)
    suffix: list[dict[int, int]] = [{} for _ in range(len(groups) + 1)]
    suffix[-1] = {k: comb(len(free_cells), k) for k in range(min(budget, len(free_cells)) + 1)}
    for j in range(len(groups) - 1, -1, -1):
        counts: dict[int, int] = defaultdict(int)
        for k, masks in groups[j][1]:
            for rest, mass in suffix[j + 1].items():
                if k + rest <= budget:
                    counts[k + rest] += len(masks) * mass
        suffix[j] = counts
    if not suffix[0].get(budget):
        return None

    flagged = (board.cells.ravel() == FLAG)
    layouts = []
    for _ in range(draws):
        layout = flagged.copy()
        remaining = budget
        for j, (cells, buckets) in enumerate(groups):
            ticket = rng.randrange(suffix[j][remaining])
            for k, masks in buckets:
                completions = suffix[j + 1].get(remaining - k, 0)
                weight = len(masks) * completions
                if ticket < weight:
                    mask = masks[ticket // completions]
                    while mask:
                        bit = mask & -mask
                        layout[cells[bit.bit_length() - 1]] = True
                        mask ^= bit
                    remaining -= k
                    break
                ticket -= weight
        if remaining:
            layout[rng.sample(free_cells, remaining)] = True
        layouts.append(layout)
    return layouts
