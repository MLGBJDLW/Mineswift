"""Screenshot -> Board.

Three pieces, one contract:

    fit_grid      geometry only: where is the lattice, in pixels
    GlyphReader   appearance only: one revealed cell crop -> (value, confidence)
    Perception    orchestration: slicing, palette learning, consensus, rejection

The contract is the one from ``ms.board``: either a Board that is right, or
``Unreadable``. A half-correct Board is worse than no Board, because the solver
believes it and one misread cell is an instant loss.

Preconditions perception cannot check for you
---------------------------------------------
* The user's rect must contain the whole grid and little else. A box that cuts the
  last cell loses only that cell's border; a box that includes a toolbar does not.
* Covered and revealed cells must look different (a theme that draws them
  identically is unreadable by construction).
* Calibrate on a frame where covered cells are the majority (right after a reset).
  ``Perception.read`` calibrates on its first frame, and that is usually the fresh
  board - which is exactly right.

Not supported yet: question-marked cells. They read as UNREADABLE rather than as
covered, because guessing "covered" would let the solver open a flagged cell.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .board import COVERED, FLAG, QUESTION, UNREADABLE, Board, Unreadable

__all__ = [
    "Rect",
    "Calibration",
    "CellReading",
    "CellReader",
    "GlyphReader",
    "Perception",
    "GridFitError",
    "fit_grid",
]

# ---------------------------------------------------------------------------
# tunables (one place, so they can be re-fitted against real screenshots)
# ---------------------------------------------------------------------------
CELL_INSET = 0.14          # fraction of the pitch skipped on each side
MIN_PITCH = 6.0            # px; below this the glyphs are illegible
MIN_QUALITY = 0.5          # line strength (x coverage) below which there is no lattice
MIN_CONTRAST = 6.0         # absolute edge gradient a real lattice produces (measured: 26-34)
CANON_H, CANON_W = 15, 11  # canonical glyph canvas for template matching

# Why the gate is 0.5 and not something clever: measured on the synthetic lattice the
# fitted borders are 1.0-1.6x the mean edge energy, and the *phase* is verified by the
# test to < 0.5 px rather than inferred from brightness. Strength is a smoke test for
# "is there a grid at all"; a threshold high enough to prove correctness there would
# reject the darker synthetic themes for no reason.


class GridFitError(Unreadable):
    """No periodic cell structure where the user said the board was."""


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    def __post_init__(self) -> None:
        if self.w <= 0 or self.h <= 0:
            raise ValueError(f"Rect must have positive size, got {self.w}x{self.h}")

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    def crop(self, frame: np.ndarray) -> np.ndarray:
        _require_frame(frame)
        H, W = frame.shape[:2]
        if self.x < 0 or self.y < 0 or self.right > W or self.bottom > H:
            raise Unreadable(
                f"rect {self} does not fit in a {W}x{H} frame "
                f"(right={self.right}, bottom={self.bottom})"
            )
        return frame[self.y:self.bottom, self.x:self.right]


@dataclass(frozen=True)
class Calibration:
    """Where the lattice is. Float pixels on purpose: 125% browser zoom is real."""

    rows: int
    cols: int
    x0: float          # frame coords of the top-left corner of cell (0, 0)
    y0: float
    pitch_x: float
    pitch_y: float
    quality: float = 0.0

    @property
    def content_size(self) -> tuple[int, int]:
        """(w, h) of the pixels we classify: inside one cell, borders excluded."""
        return (
            max(1, int(round(self.pitch_x * (1.0 - 2 * CELL_INSET)))),
            max(1, int(round(self.pitch_y * (1.0 - 2 * CELL_INSET)))),
        )

    def cell_box(self, r: int, c: int) -> tuple[int, int, int, int]:
        """(x, y, w, h) of the classified interior of cell (r, c)."""
        w, h = self.content_size
        x = self.x0 + c * self.pitch_x + self.pitch_x * CELL_INSET
        y = self.y0 + r * self.pitch_y + self.pitch_y * CELL_INSET
        return int(round(x)), int(round(y)), w, h

    def cell_center(self, r: int, c: int) -> tuple[float, float]:
        """Where the executor should click."""
        return (
            self.x0 + (c + 0.5) * self.pitch_x,
            self.y0 + (r + 0.5) * self.pitch_y,
        )


def _require_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB array, got {getattr(frame, 'shape', None)}")
    if frame.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB, got {frame.dtype}")


def _gray(rgb: np.ndarray) -> np.ndarray:
    a = rgb.astype(np.float32)
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def _edge_profiles(roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(per-column, per-row) edge strength, peaked where cell borders are.

    A border is the only edge that runs the full height of a cell, so the mean over
    the perpendicular axis (coverage) separates it from a digit stroke: measured on
    a synthetic board the border columns are 26-52 while digit columns stay around
    2. That gap is what everything downstream relies on.
    """
    g = _gray(roi)
    gy, gx = np.gradient(g)
    mag = np.sqrt(gy * gy + gx * gx)
    return mag.mean(axis=0).astype(np.float64), mag.mean(axis=1).astype(np.float64)


def _strength_profile(profile: np.ndarray) -> np.ndarray:
    """Line strength above the near-zero floor: borders 20+, strokes 0-11."""
    return np.clip(profile - np.median(profile), 0.0, None)


def _line_runs(strength: np.ndarray, threshold: float) -> list[tuple[float, float]]:
    """(centre, width) of each run of above-threshold pixels."""
    above = strength > threshold
    n = strength.size
    runs: list[tuple[float, float]] = []
    i = 0
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and above[j + 1]:
            j += 1
        runs.append((0.5 * (i + j), float(j - i + 1)))
        i = j + 1
    return runs


def _line_centres(runs: list[tuple[float, float]], gap: float) -> list[float]:
    """Merge runs that are two sides of one boundary (a bevel draws both).

    A bevel theme renders every boundary as three edges: the trailing dark bevel of the
    previous cell, the separator, and the leading light bevel of the next one. Measured,
    that produced 59 runs for 31 boundaries. They are merged into one cluster.

    The cluster's position is the midpoint of its *extremes*, not a width-weighted mean.
    The dark side is wider after downsampling (2 px vs 1 px), so a mean lands ~0.7 px
    early on every boundary - a bias that survives least squares and shows up as a
    half-pixel origin error. The two bevel edges straddle the boundary by construction,
    so their midpoint is the boundary itself.
    """
    if not runs:
        return []
    merged: list[float] = []
    first = last = runs[0][0]
    for centre, _width in runs[1:]:
        if centre - last < gap:
            last = centre
        else:
            merged.append(0.5 * (first + last))
            first = last = centre
    merged.append(0.5 * (first + last))
    return merged


def _nearest_comb(lines: np.ndarray, p0: float, n_cells: int):
    """Least-squares comb over the lines that were found.

    Returns (border, pitch, rms_residual, n_distinct_indices).
    """
    border, pitch = float(lines[0]), p0
    k = np.zeros_like(lines)
    for _ in range(3):
        k = np.round((lines - border) / pitch)
        design = np.stack([np.ones_like(k), k], axis=1)
        coef, *_ = np.linalg.lstsq(design, lines, rcond=None)
        border, pitch = float(coef[0]), float(coef[1])
    resid = lines - (border + k * pitch)
    return border, pitch, float(np.sqrt(np.mean(resid ** 2))), int(np.unique(k).size)


def _line_positions(runs: list[tuple[float, float]], profile: np.ndarray,
                    gap: float) -> np.ndarray:
    """Boundary positions: one per cluster of edges, sub-pixel.

    A boundary renders as one of three shapes depending on the theme:

        separator only   one notched column (the separator itself)
        bevel only       a dark edge and a light edge, one column apart
        separator+bevel  both, welded into one cluster

    For the two-edged shapes the boundary lies *between* the edges, so the midpoint of the
    cluster's extremes is the position. That is the rule this keeps, because it is the one
    that passed both renderers on every style except the single-notched one - and for a
    single notch the extremes collapse to the notch itself, which is correct.

    Measured residual of this rule: <=0.32 px on the pixel renderer, <=0.88 px on the
    antialiased one. The remaining error on the latter is a property of box-downsampling a
    1 px line (one column holds ~92% of the ink), not something a rule can recover; see
    `test_antialiased_theme_variants_read_the_board`, which checks that the error never
    reaches a cell crop.
    """
    if not runs:
        return np.empty(0)

    clusters: list[list[tuple[float, float]]] = [[runs[0]]]
    for run in runs[1:]:
        if run[0] - clusters[-1][-1][0] < gap:
            clusters[-1].append(run)
        else:
            clusters.append([run])

    return np.asarray([0.5 * (c[0][0] + c[-1][0]) for c in clusters], dtype=np.float64)


def fit_axis(roi: np.ndarray, axis: int, n_cells: int) -> tuple[float, float, float]:
    """(boundary, pitch, quality) for one axis: `axis=0` fits x (columns).

    boundary = outer edge of the first cell (frame coords, relative to `roi`).
    quality  = line strength x regularity; a fit whose lines are not evenly spaced is
    reported as zero so the caller rejects the board instead of slicing it slightly wrong.
    """
    prof_x, prof_y = _edge_profiles(roi)
    profile = prof_x if axis == 0 else prof_y
    strength = _strength_profile(profile)
    p0 = float(profile.size) / n_cells

    top = float(np.percentile(strength, 99))
    if top <= 0:
        raise GridFitError(
            f"no edges at all on the {'horizontal' if axis == 0 else 'vertical'} axis"
        )
    if top < MIN_CONTRAST:
        # A washed-out board still has *some* edge (measured 0.5 against 27 for a real
        # theme), and relative thresholds cannot tell the two apart. This one is absolute
        # because a real separator is bright in absolute terms, not just relative to noise.
        raise GridFitError(f'lattice contrast too low ({top:.1f} < {MIN_CONTRAST})')

    runs = _line_runs(strength, 0.45 * top)
    lines_arr = _line_positions(runs, profile, gap=max(3.0, 0.10 * p0))
    if lines_arr.size < max(4, n_cells - 2):
        raise GridFitError(
            f"found {lines_arr.size} lines on the {'horizontal' if axis == 0 else 'vertical'} "
            f"axis for {n_cells} cells; the box or the row/col count is wrong"
        )

    border, pitch, rms, n_idx = _nearest_comb(lines_arr, p0, n_cells)

    # The lattice has to lie inside the user's rect, and the comb is anchored on the first
    # line it can *see*. When the rect hugs the grid, boundary 0 sits exactly on the crop
    # edge; a bevel theme renders it as a clipped edge that can fall under the run
    # threshold. The fit then starts at boundary 1 and thirty lines describe thirty-one
    # boundaries. The rect is the missing evidence - but only if the rect really is tight.
    #
    # Measured failure of the first version of this guard: it accepted any phase whose
    # left and right overhang were each < 0.6 * pitch, which is 18 px. A phase off by
    # almost a whole cell (-29 px) therefore passed on one axis and the board shifted.
    # The condition has to be "the lattice fills this rect", not "the lattice roughly
    # fits", because the caller only asks for a tight fit in the first place.
    size = float(profile.size)
    edge_tol = min(3.0, 0.12 * abs(pitch))
    best: tuple[float, float] | None = None
    for shift in (-2, -1, 0, 1, 2):
        cand = border + shift * pitch
        left = -cand                     # gap before the first line
        right = cand + n_cells * pitch - size   # gap after the last line
        if left > edge_tol or right > edge_tol:
            continue
        cost = max(0.0, left) + max(0.0, right) + 0.01 * abs(shift)
        if best is None or cost < best[0]:
            best = (cost, cand)
    if best is not None:
        border = best[1]

    k = np.round((lines_arr - border) / pitch)
    resid = lines_arr - (border + k * pitch)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    n_idx = int(np.unique(k[(k >= 0) & (k <= n_cells)]).size)

    coverage = min(1.0, n_idx / (n_cells + 1))
    mean_line = float(profile[np.round(lines_arr).astype(int)].mean())
    quality = (mean_line / (profile.mean() + 1e-6)) * coverage
    if abs(pitch - p0) > 0.15 * p0 or rms > 1.0:
        quality = 0.0
    return border, pitch, quality


def _infer_pitch(profile: np.ndarray) -> float | None:
    """Estimate the fundamental grid period on a tightly cropped axis."""
    signal = profile - np.mean(profile)
    if np.ptp(signal) < MIN_CONTRAST:
        return None
    scores = []
    for cells in range(3, min(60, int(len(signal)/MIN_PITCH))+1):
        pitch = len(signal)/cells
        shift = round(pitch)
        left, right = signal[:-shift], signal[shift:]
        norm = np.linalg.norm(left)*np.linalg.norm(right)
        if norm:
            scores.append((float(left @ right / norm), pitch))
    if not scores:
        return None
    best = max(score for score,_ in scores)
    if best < .60:
        return None
    # A texture also repeats at twice its pitch. Prefer the shortest equally
    # well-supported period, then let fit_axis independently check the lines.
    return min(pitch for score,pitch in scores if score >= best-.025)


def fit_grid(
    frame: np.ndarray,
    rect: Rect,
    rows: int | None = None,
    cols: int | None = None,
) -> Calibration:
    """Fit a uniform lattice inside the user's rect.

    rows/cols are the reliable path; with both omitted the pitch comes from
    autocorrelation, which assumes the rect is tight around the grid.
    """
    roi = rect.crop(frame)

    if rows is None or cols is None:
        prof_x, prof_y = _edge_profiles(roi)
        if cols is None:
            p = _infer_pitch(prof_x)
            if p is None:
                raise GridFitError("cannot infer the column count; pass rows/cols")
            cols = max(1, int(round(rect.w / p)))
        if rows is None:
            p = _infer_pitch(prof_y)
            if p is None:
                raise GridFitError("cannot infer the row count; pass rows/cols")
            rows = max(1, int(round(rect.h / p)))

    rows, cols = int(rows), int(cols)
    if not (3 <= rows <= 60) or not (3 <= cols <= 60):
        raise GridFitError(f"implausible grid {rows}x{cols}; 3..60 is the supported range")

    bx, px, qx = fit_axis(roi, 0, cols)
    by, py, qy = fit_axis(roi, 1, rows)
    quality = min(qx, qy)
    if quality < MIN_QUALITY:
        raise GridFitError(
            f"no lattice inside the rect (quality {quality:.2f} < {MIN_QUALITY}); "
            "check the box and the row/col count"
        )
    return Calibration(rows, cols, rect.x + bx, rect.y + by, px, py, quality)


# ---------------------------------------------------------------------------
# glyph recognition
# ---------------------------------------------------------------------------
# A 5x7 bitmap face. The atlas holds several faces so a blocky font and a thin
# font both land close to their own template instead of halfway between two.
_FONT5X7: dict[int, tuple[str, ...]] = {
    1: ("..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."),
    2: (".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"),
    3: ("####.", "....#", "....#", ".###.", "....#", "....#", "####."),
    4: ("...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."),
    5: ("#####", "#....", "#....", "####.", "....#", "....#", "####."),
    6: ("..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."),
    7: ("#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."),
    8: (".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."),
}

# 7-segment face: a genuinely different shape, so the atlas is not one font.
_SEGMENTS = {
    "a": (0, 2, 0, 6),
    "b": (1, 8, 5, 8),
    "c": (7, 8, 11, 8),
    "d": (12, 2, 12, 6),
    "e": (7, 0, 11, 0),
    "f": (1, 0, 5, 0),
    "g": (6, 2, 6, 6),
}
_SEG7_ON = {
    1: "bc", 2: "abged", 3: "abgcd", 4: "fgbc",
    5: "afgcd", 6: "afgecd", 7: "abc", 8: "abcdefg",
}

# how many holes a digit's glyph has. A mismatch is evidence, not proof: thin
# fonts lose holes to antialiasing, so it damps confidence instead of vetoing.
_EXPECTED_HOLES = {1: {0}, 2: {0}, 3: {0}, 4: {0, 1}, 5: {0}, 6: {1}, 7: {0}, 8: {2}}

# Faces that exist in the wild but not in the two families above. A '1' is the digit with
# the most shape variation: a bare stem, a stem with a flag, and a stem on a full-width
# baseline all appear in real Minesweeper fonts. Without the baseline variant the rendered
# '1' peaked at 0.455 IoU - just above the 0.42 floor - so its confidence collapsed and
# *every* '1' on the board was rejected (measured: 0/112 accepted, 101 of the 102
# rejections in the whole sample). Adding the shape, not lowering the floor, is the fix.
_EXTRA_FACES: dict[int, tuple[tuple[str, ...], ...]] = {
    1: (
        ("..#..", ".##..", "..#..", "..#..", "..#..", "..#..", "#####"),
        ("..#..", "..#..", "..#..", "..#..", "..#..", "..#..", "..#.."),
    ),
}


def _clip01(v: float) -> float:
    return float(min(1.0, max(0.0, v)))


def _resize_nn(m: np.ndarray, oh: int, ow: int) -> np.ndarray:
    """Resize a binary mask. Shrinking must *sample*, not drop: taking every h/oh-th
    pixel of a thin stroke loses the stroke, which is exactly how a '2' became a '1'."""
    h, w = m.shape[-2:]
    if oh == h and ow == w:
        return m
    if oh >= h and ow >= w:                      # enlarge: replicate nearest
        yi = np.minimum((np.arange(oh) * h) // oh, h - 1)
        xi = np.minimum((np.arange(ow) * w) // ow, w - 1)
        return m[..., yi, :][..., :, xi]
    # shrink: max-pool each target cell over its source box, so no stroke disappears
    ys = np.linspace(0, h, oh + 1).round().astype(int)
    xs = np.linspace(0, w, ow + 1).round().astype(int)
    out = np.zeros(m.shape[:-2] + (oh, ow), dtype=m.dtype)
    for i in range(oh):
        for j in range(ow):
            block = m[..., ys[i]:max(ys[i] + 1, ys[i + 1]), xs[j]:max(xs[j] + 1, xs[j + 1])]
            out[..., i, j] = block.max(axis=(-2, -1))
    return out


def _dilate(m: np.ndarray) -> np.ndarray:
    m = m.astype(bool)
    h, w = m.shape
    p = np.pad(m, 1, constant_values=False)
    out = np.zeros((h, w), dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out |= p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    return out


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def _canon(mask: np.ndarray) -> np.ndarray | None:
    """Aspect-preserving, size-independent glyph: 15x11, centred, binary.

    Preserving the aspect ratio matters: stretching a '1' to the full width
    turns it into a '7'.
    """
    box = _bbox(mask)
    if box is None:
        return None
    y0, x0, y1, x1 = box
    sub = mask[y0:y1, x0:x1]
    h, w = sub.shape
    scale = min((CANON_H - 2) / h, (CANON_W - 2) / w)
    oh = max(1, int(round(h * scale)))
    ow = max(1, int(round(w * scale)))
    body = _resize_nn(sub, oh, ow).astype(bool)
    out = np.zeros((CANON_H, CANON_W), dtype=bool)
    ry = (CANON_H - oh) // 2
    rx = (CANON_W - ow) // 2
    out[ry:ry + oh, rx:rx + ow] = body
    return out


def _count_holes(mask: np.ndarray) -> int:
    """Background components that do not touch the border. 0->1, 6->1, 8->2 ..."""
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros_like(bg)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))

    holes = 0
    for r in range(h):
        for c in range(w):
            if bg[r, c] and not seen[r, c]:
                holes += 1
                seen[r, c] = True
                q = deque([(r, c)])
                while q:
                    rr0, cc0 = q.popleft()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        rr, cc = rr0 + dr, cc0 + dc
                        if 0 <= rr < h and 0 <= cc < w and bg[rr, cc] and not seen[rr, cc]:
                            seen[rr, cc] = True
                            q.append((rr, cc))
    return holes


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(a & b)) / union


def _build_atlas() -> dict[int, list[np.ndarray]]:
    atlas: dict[int, list[np.ndarray]] = {}
    for d in range(1, 9):
        faces = [np.array([list(row) for row in _FONT5X7[d]]) == "#"]
        canvas = np.zeros((13, 9), dtype=bool)
        for seg in _SEG7_ON[d]:
            r0, c0, r1, c1 = _SEGMENTS[seg]
            canvas[r0:r1 + 1, c0:c1 + 1] = True
        faces.append(canvas)
        for extra in _EXTRA_FACES.get(d, ()):
            faces.append(np.array([list(row) for row in extra]) == "#")
        variants: list[np.ndarray] = []
        for f in faces:
            for v in (f, _dilate(f)):
                c = _canon(v)
                if c is not None:
                    variants.append(c)
        atlas[d] = variants
    return atlas


_ATLAS = _build_atlas()


@dataclass(frozen=True)
class CellReading:
    value: int
    confidence: float
    reason: str = ""


class CellReader(Protocol):
    """The seam a trained CNN will plug into: one crop in, one reading out."""

    def read(self, crop: np.ndarray) -> CellReading: ...


def _ring_bg(crop: np.ndarray) -> np.ndarray:
    """Background colour from the outer ring of the crop.

    Per cell, not per board: a theme with a vertical gradient would otherwise
    make every glyph look like ink at one edge and like background at the other.
    """
    h, w = crop.shape[:2]
    keep = np.ones((h, w), dtype=bool)
    keep[int(0.18 * h):int(0.82 * h) + 1, int(0.18 * w):int(0.82 * w) + 1] = False
    pixels = crop[keep]
    if pixels.size == 0:
        pixels = crop.reshape(-1, crop.shape[2])
    return np.median(pixels, axis=0)


def _denoise(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    p = np.pad(mask.astype(np.uint8), 1)
    total = np.zeros((h, w), dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            total += p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    return mask & (total >= 4)


class GlyphReader:
    """Order-independent glyph reading for a single revealed cell.

    Covered/revealed is decided by Perception (it needs the board-wide palette).
    This reader only answers: blank, flag, or which digit.
    """

    def __init__(
        self,
        *,
        ink_thresh: float = 34.0,
        min_iou: float = 0.42,
        min_margin: float = 0.05,
        margin_gate: float = 0.02,   # calibrated: at 0.02 the reader accepted 99.7% of
                                     # glyphs with 0 wrong; at 0.10 it accepted 27%
        flag_thresh: float = 0.5,
    ) -> None:
        self.ink_thresh = float(ink_thresh)
        self.min_iou = float(min_iou)
        self.min_margin = float(min_margin)
        self.margin_gate = float(margin_gate)
        self.flag_thresh = float(flag_thresh)

    # -- public ------------------------------------------------------------
    def read(self, crop: np.ndarray) -> CellReading:
        bg = _ring_bg(crop)
        dist = np.sqrt(((crop.astype(np.float32) - bg) ** 2).sum(axis=2))
        ink = _denoise(dist > self.ink_thresh)
        frac = float(ink.mean())
        if frac < 0.012:
            return CellReading(0, min(1.0, 1.0 - frac / 0.012), "blank")

        flag = self._flag_score(ink)
        if flag >= self.flag_thresh:
            return CellReading(FLAG, flag, "flag:pole+blob")

        value, conf = self._digit(ink)
        return CellReading(value, conf, f"digit:template iou-margin={conf:.2f}")

    # -- internals ---------------------------------------------------------
    def _flag_score(self, mask: np.ndarray) -> float:
        """A flag is a narrow pole that spans the bottom band, under a wide blob."""
        h, w = mask.shape
        total = int(np.count_nonzero(mask))
        if total < max(8, int(0.02 * h * w)):
            return 0.0
        box = _bbox(mask)
        assert box is not None
        y0, x0, y1, x1 = box
        gh, gw = y1 - y0, x1 - x0
        if gh < 0.5 * h or gw < 3:
            return 0.0

        top = max(1, int(y0 + 0.62 * gh))
        band = mask[top:y1, x0:x1]
        if band.size == 0:
            return 0.0
        cols = band.sum(axis=0)
        peak = int(cols.max()) if cols.size else 0
        if peak < 0.75 * band.shape[0]:
            return 0.0

        wide = int(np.count_nonzero(cols >= 0.6 * peak))
        pole_frac = wide / gw
        blob = mask[y0:max(y0 + 1, int(y0 + 0.55 * gh)), x0:x1]
        coverage = float(np.count_nonzero(blob.any(axis=0))) / gw
        mass = float(blob.sum()) / total

        if pole_frac > 0.30 or coverage < 0.30 or mass < 0.45:
            return 0.0
        return _clip01(min(
            (0.30 - pole_frac) / 0.15,
            coverage / 0.55,
            mass / 0.60,
        ))

    def _digit(self, ink: np.ndarray) -> tuple[int, float]:
        """Best atlas match, with confidence from shape agreement and runner-up margin.

        Two bugs were found here by reading the fitted pixels, both worth keeping:

        * the world's '2' arrives as two stacked horizontal bands after box-downsampling
          (top bar 7 px, bottom 9 px), and '8' contains exactly that pattern; '8' scored
          highest by raw IoU until the hole-count check excluded it.
        * a candidate incompatible with its own hole count must be *rejected*, not merely
          damped: 0.7 IoU of '8' times a 0.25 penalty still outranks the true '2' at 0.70.

        So: filter by hole count, then rank only the survivors.
        """
        canon = _canon(ink)
        if canon is None:
            return 0, 0.0
        holes = _count_holes(canon)

        def compatible(d: int) -> bool:
            # a candidate may be one hole off (thin strokes lose holes at low res) but a
            # glyph with 0 holes cannot be an 8 and an 8 cannot be a 1
            return min(abs(holes - h) for h in _EXPECTED_HOLES[d]) <= 1

        scores = {
            d: max(_iou(canon, v) for v in vs)
            for d, vs in _ATLAS.items() if compatible(d)
        }
        if not scores:
            return 0, 0.0
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best_d, best = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best - runner

        if margin < self.margin_gate:
            # two digits fit almost equally well: say so instead of picking the larger
            return best_d, 0.0
        conf = (
            _clip01((best - self.min_iou) / max(1e-6, 0.8 - self.min_iou))
            * _clip01(margin / max(1e-6, self.min_margin * 2.0))
        )
        return best_d, conf


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def _split_threshold(d: np.ndarray) -> float:
    """Where the covered cluster ends, without assuming how many cells are covered.

    Covered cells are near-identical, so their distances to the prototype form a
    tight cluster with a gap after it. Take the largest gap in the lower 85%.
    """
    s = np.sort(np.asarray(d, dtype=np.float64))
    if s.size < 4:
        return max(4.0, float(s.max()) * 1.5 if s.size else 4.0)
    gaps = np.diff(s)
    lo = max(1, int(0.10 * s.size))
    hi = max(lo + 1, int(0.85 * s.size))
    k = lo + int(np.argmax(gaps[lo:hi]))
    if gaps[k] > max(6.0, 0.5 * s[k]):
        return float(s[k] + 0.5 * gaps[k])
    return float(max(4.0, 3.0 * s[int(0.5 * s.size)]))


@dataclass(eq=False)
class Palette:
    covered: np.ndarray   # (8, 8) mean-free? no: raw grey, so tone matters too
    threshold: float


class Perception:
    """Frame + user rect -> Board. Calibrates on the first frame it sees."""

    def __init__(
        self,
        rect: Rect,
        rows: int | None = None,
        cols: int | None = None,
        *,
        reader: CellReader | None = None,
        min_confidence: float = 0.30,   # absolute floor; the reader's margin gate does the work
    ) -> None:
        self.rect = rect
        self.rows = rows
        self.cols = cols
        self.reader: CellReader = reader or GlyphReader()
        self.min_confidence = float(min_confidence)
        self.calib: Calibration | None = None
        self.palette: Palette | None = None
        self.last_rejections: list[tuple[int, int, float]] = []

    # -- calibration -------------------------------------------------------
    def calibrate(self, frame: np.ndarray) -> Calibration:
        calib = fit_grid(frame, self.rect, self.rows, self.cols)
        crops = self._crops(frame, calib)
        patches = _resize_nn(_gray(crops), 8, 8)
        proto = np.median(patches, axis=0).astype(np.float32)
        d = np.sqrt(((patches - proto) ** 2).sum(axis=(1, 2)))
        threshold = _split_threshold(d)

        covered_frac = float(np.mean(d <= threshold))
        if covered_frac < 0.30:
            raise Unreadable(
                f"only {covered_frac:.0%} of cells look covered; calibrate on a fresh "
                "board (before the first click) or pass rows/cols explicitly"
            )

        self.rows, self.cols = calib.rows, calib.cols
        self.calib = calib
        self.palette = Palette(covered=proto, threshold=threshold)
        return calib

    # -- reading -----------------------------------------------------------
    def read(
        self,
        frame: np.ndarray,
        mines_left: int,
        frame_id: int = 0,
        *,
        strict: bool = True,
    ) -> Board:
        if self.calib is None or self.palette is None:
            self.calibrate(frame)
        calib = self.calib
        assert calib is not None and self.palette is not None

        if not isinstance(mines_left, (int, np.integer)):
            raise ValueError(f"mines_left must be an int, got {type(mines_left).__name__}")
        mines_left = int(mines_left)
        if not (0 <= mines_left <= calib.rows * calib.cols):
            raise ValueError(f"mines_left={mines_left} is outside 0..{calib.rows * calib.cols}")

        crops = self._crops(frame, calib)
        patches = _resize_nn(_gray(crops), 8, 8)
        d = np.sqrt(((patches - self.palette.covered) ** 2).sum(axis=(1, 2)))

        n = calib.rows * calib.cols
        cells = np.full(n, COVERED, dtype=np.int8)
        confidence = np.ones(n, dtype=np.float32)
        reasons: list[str] = [""] * n

        for k in np.flatnonzero(d > self.palette.threshold):
            reading = self.reader.read(crops[k])
            cells[k] = reading.value
            confidence[k] = reading.confidence
            reasons[k] = reading.reason

        weak = np.flatnonzero(confidence < self.min_confidence)
        self.last_rejections = [
            (int(k // calib.cols), int(k % calib.cols), float(confidence[k])) for k in weak
        ]
        if weak.size:
            cells[weak] = UNREADABLE

        board = Board(
            calib.rows,
            calib.cols,
            cells.reshape(calib.rows, calib.cols),
            mines_left,
            frame_id,
        )
        if weak.size and strict:
            sample = ", ".join(
                f"({r},{c}) p={p:.2f}" for r, c, p in self.last_rejections[:6]
            )
            raise Unreadable(
                f"{weak.size} cell(s) below confidence {self.min_confidence:.2f}: {sample}"
                + (" ..." if weak.size > 6 else "")
            )
        return board

    def read_stable(
        self,
        frames: Sequence[np.ndarray],
        mines_left: int,
        frame_id: int = 0,
    ) -> Board:
        """Two or more frames of the same moment must produce the same board.

        This is the cheap answer to animations and hover highlighting: anything
        that is still moving will disagree across frames, and disagreement is a
        rejection, not a vote.
        """
        if len(frames) < 2:
            raise ValueError("read_stable needs at least two frames")

        boards = [self.read(f, mines_left, frame_id) for f in frames]
        first = boards[0]
        for i, other in enumerate(boards[1:], start=1):
            diff = np.argwhere(first.cells != other.cells)
            if diff.size:
                r, c = (int(v) for v in diff[0])
                raise Unreadable(
                    f"frames disagree on {len(diff)} cell(s), first at ({r},{c}): "
                    f"{int(first.cells[r, c])} vs {int(other.cells[r, c])} "
                    f"(frame {0} vs {i}) - the board is probably still animating"
                )
        return first

    # -- slicing -----------------------------------------------------------
    def _crops(self, frame: np.ndarray, calib: Calibration) -> np.ndarray:
        _require_frame(frame)
        H, W = frame.shape[:2]
        _, _, cw, ch = calib.cell_box(0, 0)
        out = np.empty((calib.rows * calib.cols, ch, cw, 3), dtype=np.uint8)
        k = 0
        for r in range(calib.rows):
            for c in range(calib.cols):
                x, y, _, _ = calib.cell_box(r, c)
                # Clamp to the calibrated rect: with pitch/offset from a least-squares fit,
                # the last cell's right edge can round one pixel past the rect, which sits
                # exactly on the grid. Reading a clamped crop is always better than
                # rejecting a board whose lattice is right.
                xr = min(x, self.rect.right - cw)
                yr = min(y, self.rect.bottom - ch)
                if xr < 0 or yr < 0 or xr + cw > W or yr + ch > H:
                    raise Unreadable(
                        f"calibration puts cell ({r},{c}) at ({xr},{yr}) outside the "
                        f"{W}x{H} frame; re-fit the rect"
                    )
                out[k] = frame[yr:yr + ch, xr:xr + cw]
                k += 1
        return out
