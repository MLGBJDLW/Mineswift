"""Reproducible first-click rules that never look at a mine layout."""
from __future__ import annotations

import numpy as np

from .engine import Config


OPENINGS = ('center', 'random', 'interior', 'near-corner', 'near-corner-3', 'edge')
_OPENING_STREAM = 0x4F50454E


def _range_interior(length: int) -> range:
    return range(1, length - 1) if length > 2 else range(length)


def _near_corner(length: int, distance: int) -> tuple[int, ...]:
    # Zero-based distance in from either edge, mirrored uniformly.
    offset = min(distance, (length - 1) // 2)
    return tuple(sorted({offset, length - 1 - offset}))


def first_cell(cfg: Config, seed: int, opening: str) -> int:
    """Select a cell before generation, on a stream separate from mine placement."""
    if opening not in OPENINGS:
        raise ValueError(f'unknown opening: {opening}')
    rows, cols = cfg.rows, cfg.cols
    if opening == 'center':
        return (rows // 2) * cols + cols // 2
    rng = np.random.default_rng(np.random.SeedSequence([seed, _OPENING_STREAM]))
    if opening == 'random':
        return int(rng.integers(rows * cols))
    if opening == 'edge':
        cells = [r * cols + c for r in range(rows) for c in range(cols)
                 if r in (0, rows - 1) or c in (0, cols - 1)]
    else:
        distance = {'near-corner': 2, 'near-corner-3': 3}.get(opening)
        rr, cc = ((_near_corner(rows, distance), _near_corner(cols, distance))
                  if distance is not None else (_range_interior(rows), _range_interior(cols)))
        cells = [r * cols + c for r in rr for c in cc]
    return cells[int(rng.integers(len(cells)))]
