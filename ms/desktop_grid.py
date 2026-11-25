"""Read a user-selected rectangular lattice in physical screen coordinates."""

from pathlib import Path
import ctypes
from ctypes import wintypes
import re
import time

import numpy as np
from PIL import Image, ImageGrab

from .board import Board, COVERED, FLAG, Stuck, Unreadable
from .executor import WindowsMouse, _window_at
from .perception import Calibration, GlyphReader, GridFitError, Rect, fit_grid
from .solver import Inconsistent, NEIGHBOURS, ORTHOGONAL, raw_constraints
from .web_chrome import StopRequested, settled
from .web_vision import Frame, GridLayout


def _tile(pixels):
    import cv2

    return cv2.resize(pixels, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32)


def _axis_lines(centres, tolerance):
    values = np.sort(centres)
    groups = np.split(values, np.flatnonzero(np.diff(values) > tolerance) + 1)
    return np.array([np.median(group) for group in groups])


def locate_lattices(pixels, shapes):
    """Fit maximal rectangular groups of tile interiors, including mixed open/closed boards."""
    import cv2

    shapes = set(shapes)
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY) // 8
    height, width = gray.shape
    colors = np.argsort(np.bincount(gray.ravel(), minlength=32))[-8:]
    matches = []
    for color in colors:
        _, _, stats, _ = cv2.connectedComponentsWithStats(
            np.uint8(gray == color), connectivity=4
        )
        x, y, w, h, area = stats[1:].T
        keep = (
            (w >= 5)
            & (h >= 5)
            & (w <= 150)
            & (h <= 150)
            & (w > 0.75 * h)
            & (w < 1.33 * h)
            & (area >= 0.42 * w * h)
        )
        tiles = stats[1:][keep]
        if len(tiles) < 6:
            continue
        sizes, counts = np.unique(tiles[:, 2], return_counts=True)
        for size in sizes[counts >= 4]:
            similar = tiles[(tiles[:, 2] >= 0.73 * size) & (tiles[:, 2] <= 1.39 * size)]
            centres = similar[:, :2] + similar[:, 2:4] / 2
            # Join neighbouring centres locally. A toolbar elsewhere in the same
            # window must not add an extra row or column to a valid board.
            cx, cy = (centres / 3).astype(int).T
            mask = np.zeros((height // 3 + 1, width // 3 + 1), np.uint8)
            mask[cy, cx] = 1
            k = int(np.ceil(size * 1.65 / 3))
            mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
            _, groups = cv2.connectedComponents(mask)
            ids = groups[cy, cx]
            for group in np.unique(ids):
                pts = centres[ids == group]
                if len(pts) < 0.65 * min(r * c for r, c in shapes):
                    continue
                tol = max(1.6, size * 0.11)
                xs, ys = (_axis_lines(axis, tol) for axis in pts.T)
                rows, cols = len(ys), len(xs)
                if (rows, cols) not in shapes or len(pts) < 0.65 * rows * cols:
                    continue
                dx, dy = (axis[-1] - axis[0] for axis in (xs, ys))
                dx, dy = dx / (cols - 1), dy / (rows - 1)
                if not (size * 0.9 <= dx <= size * 1.85 and 0.91 < dx / dy < 1.1):
                    continue
                if any(
                    np.max(np.abs(axis - np.linspace(axis[0], axis[-1], len(axis))))
                    > tol
                    for axis in (xs, ys)
                ):
                    continue
                box = tuple(
                    round(v)
                    for v in (
                        xs[0] - dx / 2,
                        ys[0] - dy / 2,
                        xs[-1] + dx / 2,
                        ys[-1] + dy / 2,
                    )
                )
                if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
                    continue
                matches.append((len(pts), (rows, cols), box))
    unique = []
    for _, shape, box in sorted(matches, reverse=True):
        if not any(
            shape == other and max(abs(a - b) for a, b in zip(box, b2)) <= 3
            for other, b2 in unique
        ):
            unique.append((shape, box))
    return unique


def find_lattice(pixels, rows, cols):
    if not (3 <= rows <= 60 and 3 <= cols <= 60):
        raise ValueError("自动找盘支持 3 至 60 行和列")
    matches = locate_lattices(pixels, [(rows, cols)])
    if not matches:
        raise Unreadable("窗口中没有找到完整棋盘，可以在预览中框选边缘。")
    if len(matches) > 1:
        raise Unreadable("窗口内找到多个棋盘，请在预览中指定。")
    return matches[0][1]


def _plain_bevel(patch):
    """Identify an unmarked raised grey cell without assuming (0, 0) is covered."""
    if np.ptp(patch, axis=2).mean() >= 4 or not 130 < np.median(patch) < 252:
        return False
    if np.percentile(patch[4:-4, 4:-4], 95) - np.percentile(patch[4:-4, 4:-4], 2) > 23:
        return False
    return (
        patch[:3, 4:-4].mean() - patch[-3:, 4:-4].mean() > 38
        and patch[4:-4, :3].mean() - patch[4:-4, -3:].mean() > 38
    )


def online_face(pixels, box, cols):
    """Find the single yellow reset face, centred just above the Online grid."""
    import cv2

    left, top, right, _ = box
    pitch = (right - left) / cols
    mid = (left + right) / 2
    x, y = max(0, round(mid - pitch)), max(0, round(top - 3 * pitch))
    crop = pixels[y:top, x : round(mid + pitch)]
    if not crop.size:
        return None
    red, green, blue = crop.astype(np.int16).transpose(2, 0, 1)
    yellow = (red > 170) & (green > 110) & (red - blue > 90) & (green - blue > 50)
    _, _, stats, _ = cv2.connectedComponentsWithStats(yellow.astype(np.uint8))
    faces = [
        (x + int(xx + w / 2), y + int(yy + h / 2))
        for xx, yy, w, h, area in stats[1:]
        if 0.54 * pitch <= w <= 1.38 * pitch
        and 0.78 < w / h < 1.25
        and area > 0.38 * w * h
        and abs(x + xx + w / 2 - mid) < 0.28 * pitch
    ]
    return faces[0] if len(faces) == 1 else None


def _new_online_round(old, new):
    pattern = r"^(.*?#)\d+( - Minesweeper Online - .+)$"
    before, after = re.fullmatch(pattern, old, re.I), re.fullmatch(pattern, new, re.I)
    return bool(before and after and before.groups() == after.groups())


def _score_message(text):
    return re.fullmatch(
        r"(?:\d+ is a new (?:daily|weekly|monthly|all-time) high score! Please enter your name"
        r"|Please enter your name to submit your score \(\d+\))", text
    ) is not None


def _classic_cell(crop):
    """Windows classic colours, gated by a matching grey bevel on the reference tile."""
    pixels = crop.astype(np.int16)
    red, green, blue = pixels.transpose(2, 0, 1)
    scarlet = (red > 204) & (red - green > 140) & (red - blue > 140)
    black = pixels.max(axis=2) < 45
    if scarlet.mean() > 0.30 and black.mean() > 0.10:
        return "lost"
    red_ink = (red > 155) & (red - green > 95) & (red - blue > 85)
    if 0.02 < red_ink.mean() < 0.30 and 0.015 < black.mean() < 0.32:
        flag_y, flag_x = np.nonzero(red_ink)
        pole_y, pole_x = np.nonzero(black)
        if (
            flag_y.mean() + 0.13 * len(crop) < pole_y.mean()
            and np.ptp(flag_x) >= 2
            and pole_x.min() <= flag_x.mean() + 2
            and pole_x.max() >= flag_x.mean() - 2
        ):
            return FLAG
    # Six classic digits have a unique saturated ink colour. Flags also contain red
    # ink but have a black pole/base; ambiguous icons fall through to shape matching.
    saturation = pixels.max(axis=2) - pixels.min(axis=2)
    ink = saturation > max(65, 0.88 * saturation.max())
    if ink.sum() < 3 or (pixels.max(axis=2) < 45).sum() > 2:
        return None
    # A classic bevel establishes the skin. Within that skin use channel
    # separation: the same bright red glyph can be (255,0,0) or (234,51,35)
    # in a captured browser. Leave a brightness gap between 3/5 and 1/4.
    r, g, b = pixels[ink].T
    blue_ink = (b-r > 70) & (b-g > 70) & (abs(r-g) < 48)
    red_ink = (r-g > 70) & (r-b > 70) & (abs(g-b) < 48)
    masks = (
        blue_ink & (b >= 202), (g-r > 53) & (g-b > 65) & (g >= 85) & (g <= 165),
        red_ink & (r >= 202), blue_ink & (b >= 95) & (b <= 162),
        red_ink & (r >= 95) & (r <= 162),
        (g-r > 55) & (b-r > 55) & (abs(g-b) < 30) & (g >= 95) & (g <= 162),
    )
    for digit, mask in enumerate(masks, 1):
        if mask.mean() > 0.85:
            return digit
    return None


class DesktopReader:
    def __init__(
        self,
        frame,
        shape,
        covered,
        total,
        tiles=None,
        rules="classic",
        classic_colors=False,
        calibration=None,
    ):
        height, width = frame.shape[:2]
        self.cal = calibration if calibration is not None else self.fit(frame, shape)
        self.rows, self.cols = self.cal.rows, self.cal.cols
        cal = self.cal
        # With a tight ROI, the two selected outer boundaries constrain the pitch
        # better than a least-squares fit to the centres of asymmetric bevels.
        if (
            max(
                abs(cal.x0),
                abs(cal.y0),
                abs(cal.x0 + cal.cols * cal.pitch_x - width),
                abs(cal.y0 + cal.rows * cal.pitch_y - height),
            )
            < 1.6
        ):
            self.cal = Calibration(
                cal.rows,
                cal.cols,
                0,
                0,
                width / cal.cols,
                height / cal.rows,
                cal.quality,
            )
        if not 0 < total < self.rows * self.cols:
            raise ValueError("总雷数必须在 1 与棋盘格数之间")
        self.total = total
        self.offsets = ORTHOGONAL if rules == "orthogonal" else NEIGHBOURS
        self.glyphs = GlyphReader()
        self.templates = {}
        if tiles:
            folder = Path(tiles)
            if not folder.is_dir():
                raise ValueError("没有找到所选皮肤图块目录")
            for name, value in [
                ("covered", COVERED),
                ("flag", FLAG),
                ("hit", "lost"),
                ("mine", "lost"),
                *((str(n), n) for n in range(9)),
            ]:
                path = folder / (name + ".png")
                if path.is_file():
                    with Image.open(path) as im:
                        self.templates[value] = self.templates.get(value, []) + [
                            _tile(np.array(im.convert("RGB")))
                        ]
        if covered is None and COVERED not in self.templates and classic_colors:
            for cell in np.ndindex((self.rows, self.cols)):
                if _plain_bevel(_tile(self.crop(frame, *cell))):
                    covered = cell
                    break
        if covered is None and COVERED not in self.templates:
            # A fresh flat skin can be calibrated from identical tiles.
            ref = _tile(self.crop(frame, 0, 0))
            if all(
                np.mean((_tile(self.crop(frame, *cell)) - ref) ** 2) <= 8**2
                for cell in np.ndindex((self.rows, self.cols))
            ):
                covered = (0, 0)
        if covered is not None:
            r, c = covered
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError("请指定棋盘内的一个实际未开格")
            self.templates.setdefault(COVERED, []).append(_tile(self.crop(frame, r, c)))
        if COVERED not in self.templates:
            raise Unreadable("请在预览里点一个未开格作为皮肤样本。")
        ref = self.templates[COVERED][0]
        self.classic = (
            classic_colors
            and np.ptp(ref, axis=2).mean() < 4
            and 145 < np.median(ref) < 250
            and ref[:3, 3:-3].mean() - ref[-3:, 3:-3].mean() > 45
        )

    def crop(self, image, r, c):
        cal = self.cal
        x, y = cal.x0 + c * cal.pitch_x, cal.y0 + r * cal.pitch_y
        return image[
            max(0, round(y)) : round(y + cal.pitch_y),
            max(0, round(x)) : round(x + cal.pitch_x),
        ]

    @staticmethod
    def fit(pixels, shape):
        height, width = pixels.shape[:2]
        try:
            return fit_grid(pixels, Rect(0, 0, width, height), *shape)
        except GridFitError:
            if None in shape:
                raise
            rows, cols = shape
            padded = np.pad(pixels, ((3, 3), (3, 3), (0, 0)), mode="edge")
            box = find_lattice(padded, rows, cols)
            if max(abs(a - b) for a, b in zip(box, (3, 3, width + 3, height + 3))) > 2:
                raise Unreadable("OpenCV 检测的格线与框选边界不相符")
            return Calibration(rows, cols, 0, 0, width / cols, height / rows)

    def save_samples(self, pixels, folder, samples):
        allowed = {str(n) for n in range(9)} | {"covered", "flag", "hit", "mine"}
        destinations = []
        for label, row, col in samples:
            r, c = int(row) - 1, int(col) - 1
            if label not in allowed or not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError("图块标签或行列无效")
            path = folder / (label + ".png")
            if path.exists() or any(p == path for p, _ in destinations):
                raise ValueError(f"已有同名图块：{path.name}")
            destinations.append((path, self.crop(pixels, r, c)))
        folder.mkdir(parents=True, exist_ok=True)
        for path, pixels in destinations:
            with path.open("xb") as out:
                Image.fromarray(pixels).save(out, format="PNG")

    def decode(self, pixels, layout, previous=None):
        # Each live port holds one verified frame from this calibration. Compare
        # the entire tile, including bevel and ink, before reusing any reading.
        cells = (np.empty((self.rows, self.cols), dtype=np.int8)
                 if previous is None else previous[1].copy())
        old_pixels = None if previous is None else previous[0]
        same = old_pixels is not None and np.array_equal(pixels, old_pixels)
        unknown = []
        for r, c in (() if same else np.ndindex(cells.shape)):
            crop = self.crop(pixels, r, c)
            if old_pixels is not None and np.array_equal(crop, self.crop(old_pixels, r, c)):
                continue
            patch = _tile(crop)
            scores = sorted(
                (
                    (
                        min(float(np.mean((patch - t) ** 2)) ** 0.5 for t in examples),
                        label,
                    )
                    for label, examples in self.templates.items()
                ),
                key=lambda pair: pair[0],
            )
            distance, label = scores[0]
            if distance <= 11 and (len(scores) == 1 or scores[1][0] - distance >= 3):
                if label == "lost":
                    return Frame(layout, "lost")
                cells[r, c] = label
                continue
            if self.classic and _plain_bevel(patch):
                cells[r, c] = COVERED
                continue
            x, y, w, h = self.cal.cell_box(r, c)
            interior = pixels[max(0, y) : y + h, max(0, x) : x + w]
            classic_value = _classic_cell(interior) if self.classic else None
            if classic_value == "lost":
                return Frame(layout, "lost")
            if (
                classic_value is not None
                and classic_value not in self.templates
                and distance >= 15
            ):
                cells[r, c] = classic_value
                continue
            reading = self.glyphs.read(interior)
            if (
                distance < 15
                or reading.value < FLAG
                or reading.confidence < 0.45
                or reading.value in self.templates
            ):
                unknown.append((r + 1, c + 1))
            cells[r, c] = reading.value
        if unknown:
            raise Unreadable(
                f"无法确认 {len(unknown)} 格，首个为 {unknown[0]}；可为当前皮肤提供图块"
            )
        closed = int(np.count_nonzero(cells < 0))
        if closed < self.total:
            raise Unreadable("未开格少于总雷数，请核对皮肤和总雷数")
        if closed == self.total:
            solved = Board(
                self.rows,
                self.cols,
                np.where(cells < 0, FLAG, cells).astype(np.int8),
                0,
            )
            try:
                raw_constraints(solved, strict=True, offsets=self.offsets)
            except Inconsistent as exc:
                raise Unreadable("终局线索与剩余雷的位置不相符") from exc
            return Frame(layout, "won")
        return Frame(
            layout,
            "playing",
            Board(self.rows, self.cols, cells, self.total - int((cells == FLAG).sum())),
        )


class DesktopGame:
    name = "desktop"

    @classmethod
    def calibrated(cls, hwnd, window_rect, title, box, reader, restart_point=None):
        self = cls.__new__(cls)
        self.mouse = WindowsMouse()
        self.hwnd, self.window_rect, self.title = hwnd, window_rect, title
        self.controls = None
        self.restart_point = restart_point
        self.region = box
        x, y, _, _ = box
        c = reader.cal
        self.reader = reader
        self.layout = GridLayout(
            (
                x + c.x0,
                y + c.y0,
                x + c.x0 + c.cols * c.pitch_x,
                y + c.y0 + c.rows * c.pitch_y,
            ),
            c.rows,
            c.cols,
        )
        self.mouse.user32.GetAsyncKeyState(0x1B)
        self.mouse.user32.GetAsyncKeyState(ord("R"))
        return self

    def __init__(self, args):
        self.mouse = WindowsMouse()
        self.controls = None
        self.restart_point = args.restart_point
        sample = covered = None
        self.mouse.user32.GetAsyncKeyState(0x1B)
        if args.window:
            self.hwnd = self.find_window(args.window)
            self.window_rect, self.title = self.window_identity()
            if self.mouse.foreground_window() != self.hwnd:
                self.mouse.user32.SetForegroundWindow(self.hwnd)
            deadline = time.monotonic() + max(0, args.wait)
            while self.mouse.foreground_window() != self.hwnd:
                if time.monotonic() >= deadline:
                    raise Stuck("请将目标游戏窗口切到前台。")
                if self.mouse.user32.GetAsyncKeyState(0x1B) & 0x8001:
                    raise StopRequested("已取消。")
                time.sleep(0.08)
            view = ImageGrab.grab(
                bbox=self.window_rect, all_screens=True, include_layered_windows=True
            )
            left, top, right, bottom = find_lattice(
                np.asarray(view.convert("RGB")), *args.shape
            )
            x, y = self.window_rect[0] + left, self.window_rect[1] + top
            w, h = right - left, bottom - top
            covered = tuple(v - 1 for v in args.covered) if args.covered else None
            if covered is None and not (
                args.tiles and (args.tiles / "covered.png").is_file()
            ):
                sample = self.pick("已找到棋盘，把鼠标放在任意未开格中心，按 F8")
        elif args.region:
            x, y, w, h = args.region
            covered = tuple(v - 1 for v in args.covered) if args.covered else None
        else:
            x, y = self.pick("把鼠标放在棋盘左上边缘，按 F8")
            right, bottom = self.pick("把鼠标放在棋盘右下边缘，按 F8")
            w, h = right - x, bottom - y
            sample = self.pick("把鼠标放在任意未开格中心，按 F8")
        if w < 12 or h < 12:
            raise ValueError("框选的棋盘太小或边缘顺序有误")
        self.region = (x, y, x + w, y + h)
        at_grid = _window_at(self.mouse.user32, x + w // 2, y + h // 2)
        if args.window and at_grid != self.hwnd:
            raise Stuck("检测到的棋盘被其他窗口遮挡。")
        self.hwnd = at_grid
        self.window_rect, self.title = self.window_identity()
        self.mouse.user32.GetAsyncKeyState(ord("R"))
        self.guard()
        frame = self.capture()
        if sample is not None:
            cal = DesktopReader.fit(frame, args.shape or (None, None))
            covered = (
                int(np.floor((sample[1] - y - cal.y0) / cal.pitch_y)),
                int(np.floor((sample[0] - x - cal.x0) / cal.pitch_x)),
            )
        folder = args.tiles
        if args.sample_tile and not folder.exists():
            folder = None
        self.reader = DesktopReader(
            frame,
            args.shape or (None, None),
            covered,
            args.mines,
            folder,
            args.rules,
            args.classic_colors,
        )
        if args.sample_tile:
            self.reader.save_samples(frame, args.tiles, args.sample_tile)
        cal = self.reader.cal
        self.layout = GridLayout(
            (
                x + cal.x0,
                y + cal.y0,
                x + cal.x0 + cal.cols * cal.pitch_x,
                y + cal.y0 + cal.rows * cal.pitch_y,
            ),
            cal.rows,
            cal.cols,
        )

    def find_window(self, name):
        handles = []
        user = self.mouse.user32

        @ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
        def visit(hwnd, _):
            if user.IsWindowVisible(hwnd) and not user.IsIconic(hwnd):
                title = ctypes.create_unicode_buffer(1024)
                user.GetWindowTextW(hwnd, title, len(title))
                if title.value.casefold() == name.casefold():
                    handles.append(hwnd)
            return True

        user.EnumWindows(visit, 0)
        if len(handles) != 1:
            raise Stuck("需要唯一且未最小化的匹配窗口，请核对完整标题或用 F8 框选。")
        return handles[0]

    def pick(self, prompt):
        print(prompt + "。Esc 取消。", flush=True)
        key = self.mouse.user32.GetAsyncKeyState
        while key(0x77) & 0x8000:
            time.sleep(0.03)
        while True:
            if key(0x1B) & 0x8001:
                raise StopRequested("已取消框选。")
            if key(0x77) & 0x8000:
                return self.mouse.cursor()
            time.sleep(0.03)

    def window_identity(self):
        user = self.mouse.user32
        rect = wintypes.RECT()
        title = ctypes.create_unicode_buffer(1024)
        user.GetWindowTextW(self.hwnd, title, len(title))
        if not user.GetWindowRect(self.hwnd, ctypes.byref(rect)):
            raise Stuck("选定的窗口已关闭。")
        return (rect.left, rect.top, rect.right, rect.bottom), title.value

    def guard(self):
        user = self.mouse.user32
        if user.GetAsyncKeyState(0x1B) & 0x8001 or (
            self.controls and self.controls.stop.is_set()
        ):
            raise StopRequested("已停止。")
        rect, title = self.window_identity()
        if self.mouse.foreground_window() != self.hwnd or rect != self.window_rect:
            raise Stuck("目标窗口、标题或位置已经改变，请重新框选。")
        if title != self.title:
            if not getattr(self, "restarting", False) or not _new_online_round(
                self.title, title
            ):
                raise Stuck("游戏窗口标题已改变，请重新识别游戏。")
            self.title = title
        if any(user.GetAsyncKeyState(v) & 0x8000 for v in (1, 2, 4)):
            raise Stuck("检测到用户鼠标按键，已停止。")
        if self.controls and (
            self.controls.again.is_set() or user.GetAsyncKeyState(ord("R")) & 1
        ):
            from .web_chrome import RestartRound

            self.controls.again.clear()
            raise RestartRound

    def capture(self):
        self.guard()
        x, y, r, b = self.region
        points = [
            (x + 1, y + 1),
            (x + 1, b - 2),
            (r - 2, y + 1),
            (r - 2, b - 2),
            ((x + r) // 2, (y + b) // 2),
        ]
        if hasattr(self, "layout"):
            points.extend(
                self.layout.center(cell)
                for cell in np.ndindex((self.layout.rows, self.layout.cols))
            )
        for px, py in points:
            if _window_at(self.mouse.user32, px, py) != self.hwnd:
                raise Stuck("棋盘被其他窗口遮挡。")
        # Pillow on Windows grabs all monitors before cropping. GDI can copy
        # just this rectangle, including layered windows, in physical pixels.
        try:
            from mss import MSS
            from mss.exception import ScreenShotError
        except ImportError:
            pass
        else:
            import cv2
            try:
                with MSS() as screen:
                    return cv2.cvtColor(np.asarray(screen.grab(self.region)), cv2.COLOR_BGRA2RGB)
            except (OSError, ScreenShotError):
                pass
        return np.asarray(
            ImageGrab.grab(
                bbox=self.region, all_screens=True, include_layered_windows=True
            ).convert("RGB")
        )

    def read(self):
        self.last_pixels = None
        self.last_pixels = self.capture()
        previous = getattr(self, "_verified", None)
        frame = self.reader.decode(self.last_pixels, self.layout, previous)
        if frame.board is not None and "minesweeper online" in self.title.casefold():
            cells = frame.board.cells
            padded = np.pad(cells == COVERED, 1)
            for dr, dc in NEIGHBOURS:
                if np.any(
                    (cells == 0)
                    & padded[
                        1 + dr : 1 + dr + frame.board.rows,
                        1 + dc : 1 + dc + frame.board.cols,
                    ]
                ):
                    raise Unreadable("空白格旁仍有未展开的格子，正在等待网页完成落子")
        self._verified = ((self.last_pixels.copy(), frame.board.cells.copy())
                          if frame.board is not None else None)
        return frame

    def score_prompt(self):
        """Recognize Chrome's native score prompt on the bound .com window."""
        if not self.title.startswith("Minesweeper Online - Play Free Online Minesweeper"):
            return None
        from comtypes import COMError
        from comtypes.client import CreateObject, GetModule

        try:
            u = GetModule("UIAutomationCore.dll")
            ui = CreateObject(u.CUIAutomation, interface=u.IUIAutomation)
            edit = ui.GetFocusedElement()
            if not edit or edit.CurrentControlType != u.UIA_EditControlTypeId or not _score_message(edit.CurrentName):
                return None
            root = ui.ElementFromHandle(self.hwnd)
            node = edit
            modal = origin = None
            for _ in range(12):
                if not node:
                    break
                if ui.CompareElements(node, root):
                    if modal and origin:
                        return ui, u, origin
                    break
                if node.CurrentClassName == "JavaScriptTabModalDialogViewViews":
                    modal = node
                if modal and node.CurrentControlType == u.UIA_WindowControlTypeId and re.match(
                    r"^(?:https://)?(?:www\.)?minesweeperonline\.com(?:\s|$)", node.CurrentName, re.I
                ):
                    origin = node
                node = ui.ControlViewWalker.GetParentElement(node)
        except (COMError, OSError):
            pass
        return None

    def covered_terminal(self):
        if self.score_prompt() is not None:
            self.guard()
            return Frame(self.layout, "won")
        return None

    def press(self, point, button):
        self.guard()
        if _window_at(self.mouse.user32, *point) != self.hwnd:
            raise Stuck("点击位置不属于绑定的游戏窗口。")
        self.mouse.move(*point)
        self.guard()
        if (
            self.mouse.cursor() != point
            or _window_at(self.mouse.user32, *point) != self.hwnd
        ):
            raise Stuck("点击位置发生变化。")
        with self.mouse.grabbed():
            if button == "chord":
                self.mouse.pressed_together(("left", "right"))
                time.sleep(0.012)
                self.mouse.released_together(("right", "left"))
            else:
                self.mouse.press(button)
                time.sleep(0.012)
                self.mouse.release(button)
        # Leave the pointer here. Readback observes mouseup and the new board.

    def click(self, before, cell, button):
        if not self.read().matches(before):
            raise Stuck("落子前棋盘发生变化。")
        self.press(self.layout.center(cell), button)

    def restart(self):
        if self.restart_point is None:
            raise Stuck(
                "此窗口未设置重开按钮。可用 --restart-point X Y 指定其屏幕坐标。"
            )
        prompt = self.score_prompt()
        if prompt:
            ui, u, dialog = prompt
            buttons = dialog.FindAll(4, ui.CreatePropertyCondition(u.UIA_ControlTypePropertyId, u.UIA_ButtonControlTypeId))
            cancel = next((b for i in range(buttons.Length) if (b := buttons.GetElement(i)).CurrentName in ("取消", "Cancel")), None)
            if not cancel:
                raise Stuck("网页正在登记成绩。请取消成绩弹窗后重开。")
            self.guard()
            cancel.GetCurrentPattern(u.UIA_InvokePatternId).QueryInterface(u.IUIAutomationInvokePattern).Invoke()
        before = settled(self)
        if before.board is not None and before.board.covered().all():
            return before
        self.press(tuple(self.restart_point), "left")
        self.restarting = True
        try:
            after = settled(self, before, timeout=6)
        finally:
            self.restarting = False
        if after.board is None or not after.board.covered().all():
            raise Stuck("重开后没有识别到全新棋盘。")
        return after
