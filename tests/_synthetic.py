"""Synthetic board renderer: the world the perception tests measure against.

Deliberately knows nothing about ``ms.perception`` - it only knows how to draw a
cell and what the true value of that cell is. Everything is rendered at 3x and
box-downsampled, so glyph edges are genuinely antialiased rather than a
pixel-perfect copy of any classifier template.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

COVERED, FLAG = -1, -2

# The world's 5x7 face. Similar spirit to any blocky minesweeper font, but not
# the atlas's face: '1' carries a serif, '7' has a longer stem.
FONT: dict[int, tuple[str, ...]] = {
    1: ("...#.", "..##.", ".#.#.", "...#.", "...#.", "...#.", "..###"),
    2: (".###.", "#...#", "....#", "..##.", ".#...", "#....", "#####"),
    3: ("####.", "....#", "....#", ".###.", "....#", "....#", "####."),
    4: ("...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."),
    5: ("#####", "#....", "####.", "....#", "....#", "....#", "####."),
    6: (".###.", "#....", "#....", "####.", "#...#", "#...#", ".###."),
    7: ("#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."),
    8: (".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."),
}

SEGMENTS = {
    "a": (0, 2, 0, 6), "b": (1, 8, 5, 8), "c": (7, 8, 11, 8), "d": (12, 2, 12, 6),
    "e": (7, 0, 11, 0), "f": (1, 0, 5, 0), "g": (6, 2, 6, 6),
}
SEG7_ON = {
    1: "bc", 2: "abged", 3: "abgcd", 4: "fgbc",
    5: "afgcd", 6: "afgecd", 7: "abc", 8: "abcdefg",
}


@dataclass(frozen=True)
class Theme:
    cell: int
    covered: tuple[int, int, int]
    covered_light: tuple[int, int, int]
    covered_dark: tuple[int, int, int]
    bg: tuple[int, int, int]
    ink: tuple[int, int, int]
    grid: tuple[int, int, int]
    flag: tuple[int, int, int]
    bevel: bool = True
    separator: bool = True
    margin: int = 12
    face: str = "block"      # "block" | "seg7" | "wide"


@dataclass
class Scene:
    frame: np.ndarray          # mid-game
    fresh: np.ndarray          # same geometry, everything covered (calibration frame)
    cells: np.ndarray
    rect: "Rect"
    x0: float
    y0: float
    pitch: float
    mines_left: int = 0


def _clip_rgb(v) -> tuple[int, int, int]:
    a = np.clip(np.asarray(v, dtype=np.float64), 0, 255).astype(int)
    return int(a[0]), int(a[1]), int(a[2])


def random_theme(rng: np.random.Generator, cell: int | None = None, face: str = "block") -> Theme:
    cell = int(cell if cell is not None else rng.choice([18, 20, 21, 24, 25, 28, 30, 32]))
    dark = bool(rng.random() < 0.3)
    if dark:
        bg = rng.integers(18, 58, size=3)
        covered = bg + rng.integers(52, 92, size=3)
    else:
        bg = rng.integers(206, 249, size=3)
        covered = bg - rng.integers(52, 92, size=3)
    covered = np.clip(covered, 0, 255)
    ink = 255 - bg
    grid = np.clip(covered - rng.integers(45, 75), 0, 255)
    flag = np.array([220, rng.integers(30, 80), 40])
    return Theme(
        cell=cell,
        covered=_clip_rgb(covered),
        covered_light=_clip_rgb(covered + 28),
        covered_dark=_clip_rgb(covered - 30),
        bg=_clip_rgb(bg),
        ink=_clip_rgb(ink),
        grid=_clip_rgb(grid),
        flag=_clip_rgb(flag),
        bevel=bool(rng.random() < 0.75),
        separator=bool(rng.random() < 0.85),
        margin=int(rng.choice([10, 12, 16])),
        face=face,
    )


def _draw_face(img, x, y, size, digit, theme: Theme, ss: int) -> None:
    if theme.face == "seg7":
        canvas = np.zeros((13, 9), dtype=bool)
        for seg in SEG7_ON[digit]:
            r0, c0, r1, c1 = SEGMENTS[seg]
            canvas[r0:r1 + 1, c0:c1 + 1] = True
        gh, gw = canvas.shape
        stretch = 1
    else:
        canvas = np.array([list(row) for row in FONT[digit]]) == "#"
        gh, gw = canvas.shape           # 7 x 5
        stretch = 2 if theme.face == "wide" else 1

    s = max(1, int(0.62 * size) // gh)
    sx, sy = s * stretch, s
    gx = x + (size - gw * sx) // 2
    gy = y + (size - gh * sy) // 2
    ys, xs = np.nonzero(canvas)
    for i, j in zip(ys, xs):
        img[gy + i * sy:gy + (i + 1) * sy, gx + j * sx:gx + (j + 1) * sx] = theme.ink


def _draw_cell(img, x, y, size, value, theme: Theme, ss: int) -> None:
    """Draw one cell into the 3x supersampled canvas at (x, y) with side `size`.

    The cell covers its whole pitch and draws its *own* trailing separator, so the
    lattice lines land at x + size for every cell. Drawing a line at size-ss and
    leaving the last pixel to the background would shift the rendered pitch by one
    pixel per cell relative to `theme.cell` - the classic off-by-one that makes a
    grid fitter report lines "between" cells.
    """
    b = max(1, ss)
    if value in (COVERED, FLAG):
        img[y:y + size, x:x + size] = theme.covered
        if theme.bevel:
            img[y:y + b, x:x + size] = theme.covered_light
            img[y:y + size, x:x + b] = theme.covered_light
            img[y + size - b:y + size, x:x + size] = theme.covered_dark
            img[y:y + size, x + size - b:x + size] = theme.covered_dark

        if value == FLAG:
            pole_w = max(1, size // 14)
            px = x + int(0.64 * size)
            img[y + int(0.18 * size):y + int(0.86 * size), px:px + pole_w] = theme.flag
            top, bot = y + int(0.20 * size), y + int(0.54 * size)
            right, left = px, x + int(0.24 * size)
            for yy in range(top, bot):
                t = (yy - top) / max(1, bot - top)
                xx = int(round(right - (right - left) * t))
                img[yy:yy + 1, xx:right] = theme.flag
    else:
        img[y:y + size, x:x + size] = theme.bg
        if value > 0:
            _draw_face(img, x, y, size, int(value), theme, ss)

    if theme.separator:
        img[y:y + size, x:x + 1] = theme.grid
        img[y:y + 1, x:x + size] = theme.grid


def sweep(cells: np.ndarray, theme: Theme) -> tuple[np.ndarray, tuple[int, int]]:
    """Render cell by cell, no supersampling. The canonical frame the tests use.

    Cell (r, c) owns the square [m + c*p, m + (c+1)*p) and draws its own trailing
    border, so every lattice line is exactly 1 px at frame position `m + c*p`.
    No resampling, so a test can assert the geometry to the pixel.
    """
    rows, cols = cells.shape
    p, m = theme.cell, theme.margin
    W, H = cols * p + 2 * m, rows * p + 2 * m
    img = np.full((H, W, 3), theme.bg, dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            _draw_cell(img, m + c * p, m + r * p, p, int(cells[r, c]), theme, 1)
    return img, (m, m)


def render(cells: np.ndarray, theme: Theme, *, ss: int = 3) -> tuple[np.ndarray, tuple[int, int]]:
    rows, cols = cells.shape
    p, m = theme.cell, theme.margin
    W, H = cols * p + 2 * m, rows * p + 2 * m

    img = np.empty((H * ss, W * ss, 3), dtype=np.uint8)
    img[:] = theme.grid
    for r in range(rows):
        for c in range(cols):
            _draw_cell(
                img,
                (m + c * p) * ss,
                (m + r * p) * ss,
                p * ss,
                int(cells[r, c]),
                theme,
                ss,
            )
    if ss > 1:
        img = img.reshape(H, ss, W, ss, 3).mean(axis=(1, 3)).astype(np.uint8)
    return img, (m, m)


def count_neighbours(mines: np.ndarray) -> np.ndarray:
    rows, cols = mines.shape
    out = np.zeros((rows, cols), np.int8)
    pad = np.pad(mines.astype(np.int8), 1)
    for dr in range(3):
        for dc in range(3):
            if dr == 1 and dc == 1:
                continue
            out += pad[dr:dr + rows, dc:dc + cols]
    return out


def random_state(
    rng: np.random.Generator,
    rows: int = 16,
    cols: int = 30,
    mines: int = 99,
    *,
    extra_open: int = 40,
    flags: int = 2,
) -> tuple[np.ndarray, int]:
    """A plausible mid-game board: flooded hole, some digits, a couple of flags."""
    lay = np.zeros(rows * cols, dtype=bool)
    lay[rng.choice(rows * cols, size=mines, replace=False)] = True
    mines2d = lay.reshape(rows, cols)
    adj = count_neighbours(mines2d)
    cells = np.full((rows, cols), COVERED, np.int8)

    safe = np.argwhere(~mines2d)
    start = tuple(int(v) for v in safe[rng.integers(len(safe))])
    stack = [start]
    while stack:
        r, c = stack.pop()
        if cells[r, c] != COVERED or mines2d[r, c]:
            continue
        cells[r, c] = adj[r, c]
        if adj[r, c] == 0:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols and cells[rr, cc] == COVERED:
                        stack.append((rr, cc))

    for _ in range(extra_open):
        r, c = (int(v) for v in safe[rng.integers(len(safe))])
        if cells[r, c] == COVERED:
            cells[r, c] = adj[r, c]

    if flags:
        flagged = 0
        for r, c in np.argwhere(mines2d):
            if flagged >= flags:
                break
            if cells[r, c] == COVERED:
                cells[r, c] = FLAG
                flagged += 1
    return cells, mines - int(np.count_nonzero(cells == FLAG))


def make_scene(
    rng: np.random.Generator,
    rows: int = 16,
    cols: int = 30,
    mines: int = 99,
    *,
    cell: int | None = None,
    face: str = "block",
    theme: Theme | None = None,
) -> Scene:
    th = theme or random_theme(rng, cell=cell, face=face)
    cells, left = random_state(rng, rows, cols, mines)
    fresh = np.full((rows, cols), COVERED, np.int8)
    fresh_img, (mx, my) = render(fresh, th)
    mid_img, _ = render(cells, th)

    from ms.perception import Rect  # the only thing the world borrows, for the return type

    return Scene(
        frame=mid_img,
        fresh=fresh_img,
        cells=cells,
        rect=Rect(mx, my, cols * th.cell, rows * th.cell),
        x0=float(mx),
        y0=float(my),
        pitch=float(th.cell),
        mines_left=left,
    )
