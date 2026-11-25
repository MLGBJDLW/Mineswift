"""Pure-python/rules-level Minesweeper simulator. This is the ground truth used by the
offline evaluator; it is deliberately independent of the solver so a solver bug cannot
mask itself."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .board import COVERED, FLAG, Board

NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass(frozen=True)
class Config:
    rows: int = 16
    cols: int = 30
    mines: int = 99
    first_click_safe: bool = True  # modern minesweeper: first click opens a 9-cell hole


BEGINNER = Config(9, 9, 10)
INTERMEDIATE = Config(16, 16, 40)
EXPERT = Config(16, 30, 99)


def neighbour_ids(rows: int, cols: int) -> np.ndarray:
    """(rows*cols, 8) neighbour index table, with -1 where off-board."""
    out = np.full((rows * cols, 8), -1, dtype=np.int32)
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            k = 0
            for dr, dc in NEIGHBOURS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    out[i, k] = rr * cols + cc
                    k += 1
    return out


class Minefield:
    """Bit-exact simulator. `visible` mirrors what the player can see:
    -1 covered, -2 flagged, 0..8 revealed."""

    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.n = cfg.rows * cfg.cols
        self.nb = neighbour_ids(cfg.rows, cfg.cols)
        self.mines = np.zeros(self.n, dtype=bool)
        self.visible = np.full(self.n, -1, dtype=np.int8)
        self.placed = False
        self.exploded = False
        self.won = False

    # -- rules -------------------------------------------------------------
    def _place(self, safe_idx: int) -> None:
        free = np.ones(self.n, dtype=bool)
        if self.cfg.first_click_safe:
            free[safe_idx] = False
            free[self.nb[safe_idx][self.nb[safe_idx] >= 0]] = False
        idx = np.flatnonzero(free)
        pick = self.rng.choice(idx, size=self.cfg.mines, replace=False)
        self.mines[pick] = True
        self.adjacent = self._adjacent_counts()
        self.placed = True

    def _adjacent_counts(self) -> np.ndarray:
        m = self.mines.astype(np.int32)
        cnt = np.zeros(self.n, dtype=np.int32)
        for k in range(8):
            ids = self.nb[:, k]
            ok = ids >= 0
            cnt[ok] += m[ids[ok]]
        return cnt

    def load(self, mines: np.ndarray) -> None:
        """Install a mine layout directly (deterministic tests and replays).

        Takes either the flat (n,) array the simulator uses internally or the 2D
        (rows, cols) array a reader would produce, because both show up at call sites and
        silently accepting the wrong shape is how a test ends up asserting the wrong board.
        """
        m = np.asarray(mines)
        if m.shape == (self.cfg.rows, self.cfg.cols):
            m = m.reshape(-1)
        if m.shape != (self.n,):
            raise ValueError(
                f"mines shape {m.shape} != {(self.n,)} flat or "
                f"{(self.cfg.rows, self.cfg.cols)} 2D"
            )
        self.mines = m.astype(bool).copy()
        self.adjacent = self._adjacent_counts()
        self.placed = True

    def _flood(self, start: int) -> None:
        stack = [start]
        while stack:
            i = stack.pop()
            if self.visible[i] >= 0:
                continue
            self.visible[i] = self.adjacent[i]
            if self.adjacent[i] == 0:
                for j in self.nb[i]:
                    if j >= 0 and self.visible[j] < 0:
                        stack.append(int(j))

    def flag(self, idx: int) -> None:
        """Toggle a flag (solver output -> executor action)."""
        if self.won or self.exploded or self.visible[idx] >= 0:
            return
        self.visible[idx] = FLAG if self.visible[idx] != FLAG else COVERED

    def click(self, idx: int) -> str:
        """Reveal a cell. Returns 'ok' | 'boom' | 'win' | 'noop'."""
        if self.won or self.exploded:
            return "noop"
        if self.visible[idx] >= 0:
            return "noop"
        if not self.placed:
            self._place(idx)
        if self.mines[idx]:
            self.exploded = True
            return "boom"
        self._flood(idx)
        self._check_win()
        return "win" if self.won else "ok"

    def chord(self, idx: int) -> str:
        """Both-button (or middle) click on a revealed number.

        Legal only when the flags around it equal the number; then every other covered
        neighbour opens at once. Returns 'ok' | 'boom' | 'win' | 'noop'.
        """
        if self.won or self.exploded:
            return "noop"
        v = int(self.visible[idx])
        if v < 0:
            return "noop"
        nbrs = [int(j) for j in self.nb[idx] if j >= 0]
        if sum(1 for j in nbrs if int(self.visible[j]) == FLAG) != v:
            return "noop"
        targets = [j for j in nbrs
                   if int(self.visible[j]) < 0 and int(self.visible[j]) != FLAG]
        if not targets:
            return "noop"
        for j in targets:
            if self.mines[j]:
                self.exploded = True
                return "boom"
            self._flood(j)
        self._check_win()
        return "win" if self.won else "ok"

    def _check_win(self) -> None:
        if np.all((self.visible >= 0) | self.mines):
            self.won = True

    def total_safe(self) -> int:
        return self.n - self.cfg.mines

    # -- what the solver is allowed to see ---------------------------------
    def observation(self) -> np.ndarray:
        return self.visible.reshape(self.cfg.rows, self.cfg.cols).copy()

    def board(self, frame_id: int = 0) -> Board:
        """Exactly what a screen reader would hand the solver."""
        return Board(
            self.cfg.rows,
            self.cfg.cols,
            self.visible.reshape(self.cfg.rows, self.cfg.cols).copy(),
            self.mines_left_counter(),
            frame_id,
        )

    def mines_left_counter(self) -> int:
        flags = int(np.sum(self.visible == -2))
        return self.cfg.mines - flags
