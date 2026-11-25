"""Read the classic minesweeper.cn canvas, face and counter from desktop pixels."""
from __future__ import annotations

import base64
import io
import os
import re
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .board import Board, COVERED, FLAG, Unreadable

Rect = tuple[int, int, int, int]
SPRITES = "https://www.minesweeper.cn/gfs1.js"


def _features(image):
    return np.asarray(image.convert("RGB").resize((16, 16), Image.Resampling.BILINEAR), dtype=np.float32)


class Templates:
    def __init__(self, images):
        self.labels = tuple(images)
        self.templates = np.stack([_features(images[k]) for k in self.labels])

    def read(self, image, alternatives=()):
        distances = np.min([np.sqrt(np.mean((_features(view) - self.templates) ** 2, axis=(1, 2, 3)))
                            for view in (image, *alternatives)], axis=0)
        a, b = np.argsort(distances)[:2]
        if distances[a] > 37 or distances[b] - distances[a] < 3:
            raise Unreadable(f"图案不清晰（匹配误差 {distances[a]:.1f}，次近 {distances[b]:.1f}）")
        return self.labels[a]


class Sprites:
    def __init__(self, data):
        # Parse only the static image literals. Page code is never evaluated.
        images = {}
        for group, idx, encoded in re.findall(
            rb'(sgf|bgf|fgf|dgf)\[(\d+)\](?:\.src)?="data:img/gif;base64,([^"]+)"', data
        ):
            expected = {b"sgf": (25, 25), b"bgf": (25, 25), b"fgf": (21, 21), b"dgf": (13, 23)}
            with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as image:
                if image.format != "GIF" or image.size != expected[group]:
                    raise ValueError("网页图块尺寸已变更")
                images[group, int(idx)] = image.convert("RGB")
        tiles = {n: images[b"bgf", n] for n in range(9)}
        tiles.update({COVERED: images[b"sgf", 0], FLAG: images[b"sgf", 1],
                      "hit": images[b"sgf", 2], "mine": images[b"sgf", 3]})
        self.tiles = Templates(tiles)
        self.faces = Templates({name: images[b"fgf", i] for i, name in enumerate(("playing", "won", "lost"))})
        self.digits = Templates({n: images[b"dgf", n] for n in range(10)})

    @classmethod
    def load(cls):
        folder = Path(os.getenv("LOCALAPPDATA", Path.home() / ".cache")) / "Mineswift"
        path = folder / "classic-sprites-v1.txt"
        if path.exists():
            return cls(path.read_bytes())
        with urllib.request.urlopen(SPRITES, timeout=12) as response:
            data = response.read(256_001)
        if len(data) > 256_000:
            raise ValueError("网页图块资源过大")
        result = cls(data)
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=folder, delete=False) as tmp:
            tmp.write(data)
        os.replace(tmp.name, path)
        return result


@dataclass(frozen=True)
class Layout:
    grid: Rect
    face: Rect
    counter: Rect

    def geometry(self):
        x, y, right, bottom = self.grid
        width, height = right - x, bottom - y
        counter_height = self.counter[3] - self.counter[1]
        estimate = counter_height / 23
        if not 0.45 <= estimate <= 5:
            raise Unreadable("计数器尺寸异常")
        # UIA rounds rectangles outwards. A one-pixel error on the small counter
        # would otherwise remove an entire column on a zoomed-out expert board.
        def possible(span):
            lo = max(1, int((span / ((counter_height + 2)/23) - 4)/25))
            hi = min(150, int((span / ((counter_height - 2)/23) - 4)/25) + 1)
            return range(lo, hi + 1)

        fw, fh = self.face[2] - self.face[0], self.face[3] - self.face[1]
        cw = self.counter[2] - self.counter[0]
        digits = max(1, round(cw / (13*estimate)))
        best = None
        for rows in possible(height):
            for cols in possible(width):
                sx, sy = width / (25*cols + 4), height / (25*rows + 4)
                errors = (abs(sx-sy)*25*min(rows, cols), fw-25*sx, fh-25*sy,
                          cw-13*digits*sx, counter_height-23*sy)
                if max(map(abs, errors)) > 2.75:
                    continue
                fit = sum(e*e for e in errors)
                if best is None or fit < best[0]:
                    best = (fit, rows, cols, sx, sy)
        if best is None:
            raise Unreadable("棋盘与计数器缩放不一致")
        return best[1:]

    def center(self, cell):
        _, _, sx, sy = self.geometry()
        r, c = cell
        return (round(self.grid[0] + (14.5 + 25*c)*sx), round(self.grid[1] + (14.5 + 25*r)*sy))

    def bounds(self):
        boxes = self.grid, self.face, self.counter
        return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))


@dataclass
class Frame:
    layout: Layout
    status: str
    board: Board | None = None

    def matches(self, other):
        if self.layout != other.layout or self.status != other.status:
            return False
        if self.board is None or other.board is None:
            return self.board is other.board
        return self.board.mines_left == other.board.mines_left and np.array_equal(self.board.cells, other.board.cells)


@dataclass(frozen=True)
class GridLayout:
    grid: Rect
    rows: int
    cols: int

    def bounds(self):
        return self.grid

    def geometry(self):
        return self.rows, self.cols

    def center(self, cell):
        x, y, right, bottom = self.grid
        r, c = cell
        return (round(x + (c+.5)*(right-x)/self.cols), round(y + (r+.5)*(bottom-y)/self.rows))


def decode(image, layout, sprites):
    """image is a screenshot cropped to layout.bounds(), in physical pixels."""
    ox, oy, _, _ = layout.bounds()

    def crop(x1, y1, x2, y2):
        return image.crop((round(x1) - ox, round(y1) - oy, round(x2) - ox, round(y2) - oy))

    def read_indicator(templates, x1, y1, x2, y2):
        # Accessibility encloses fractional CSS pixels. Allow a one-pixel raster offset.
        sample = crop(x1, y1, x2, y2)
        nearby = [crop(x1+i, y1+j, x2+i, y2+j) for i in (-1,0,1) for j in (-1,0,1) if i or j]
        try:
            return templates.read(sample, nearby)
        except Unreadable:
            # Fractional window origins can also include an extra border pixel.
            nearby.extend(crop(x1+i+k, y1+j+k, x2+i-k, y2+j-k)
                          for k in (-1,1) for i in (-1,0,1) for j in (-1,0,1))
            return templates.read(sample, nearby)

    x, y, right, bottom = layout.face
    dx, dy = (right-x) * 2/25, (bottom-y) * 2/25
    status = read_indicator(sprites.faces, x+dx, y+dy, right-dx, bottom-dy)
    if status != "playing":
        return Frame(layout, status)
    rows, cols, sx, sy = layout.geometry()
    x, y, right, bottom = layout.counter
    n = round((right-x) / (13*sx))
    if not 1 <= n <= 5:
        raise Unreadable("无法确定剩余雷数的位数")
    width = (right-x) / n
    counter = 0
    for i in range(n):
        counter = counter * 10 + read_indicator(sprites.digits, x+i*width, y, x+(i+1)*width, bottom)
    gx, gy, _, _ = layout.grid
    cells = np.empty((rows, cols), dtype=np.int8)
    for r, c in np.ndindex(rows, cols):
        x, y = gx + (2 + 25*c)*sx, gy + (2 + 25*r)*sy
        try:
            value = sprites.tiles.read(crop(x, y, x + 25*sx, y + 25*sy))
            if not isinstance(value, int):
                raise Unreadable("表情与雷格状态不一致")
        except Unreadable as exc:
            raise Unreadable(f"第 {r+1} 行第 {c+1} 列：{exc}") from exc
        cells[r, c] = value
    if counter > int((cells == COVERED).sum()):
        raise Unreadable("剩余雷数超过未标记格数")
    return Frame(layout, status, Board(rows, cols, cells, counter))
