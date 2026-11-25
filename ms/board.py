"""The single data contract between layers.

Perception produces a Board, the solver consumes it, the executor consumes a Move. Nothing
else crosses a layer boundary, so a bug is always attributable to exactly one layer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COVERED = -1      # untouched cell
FLAG = -2         # flagged by the player
QUESTION = -3     # question mark (treated as covered, not as a mine)
UNREADABLE = -4   # perception could not classify this cell


@dataclass(frozen=True)
class Board:
    rows: int
    cols: int
    cells: np.ndarray   # int8, shape (rows, cols), values: COVERED/FLAG/QUESTION/0..8
    mines_left: int     # remaining-mine counter; cross-checks, never logic
    frame_id: int = 0

    def __post_init__(self) -> None:
        if self.cells.shape != (self.rows, self.cols):
            raise ValueError(f"cells shape {self.cells.shape} != {(self.rows, self.cols)}")
        if self.cells.dtype != np.int8:
            raise ValueError(f"cells dtype must be int8, got {self.cells.dtype}")

    def covered(self) -> np.ndarray:
        """Cells the solver may open: covered or question-marked."""
        return (self.cells == COVERED) | (self.cells == QUESTION)

    def flags(self) -> np.ndarray:
        return self.cells == FLAG


class Unreadable(Exception):
    """Perception could not produce a trustworthy Board. Never emit a half-correct one."""


class Stuck(Exception):
    """A move was applied but the screen never settled, or the game state is unknown."""
