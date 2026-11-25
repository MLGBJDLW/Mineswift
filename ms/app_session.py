"""One connection and one worker for the desktop application's capture/preview/play loop."""

from contextlib import ExitStack
import ctypes
from ctypes import wintypes
from datetime import datetime
import json
import os
from pathlib import Path
import secrets
import time

import numpy as np
from PIL import Image, ImageGrab

from .board import Stuck, Unreadable
from .desktop_grid import DesktopGame, DesktopReader, locate_lattices, online_face
from .play_web import run_rounds, visible_rows
from .perception import Calibration
from .solver import Inconsistent
from .web_chrome import ChromeGame, ReadbackError, StopRequested, settled
from .web_vision import Sprites

PRESETS = {(9, 9): 10, (8, 8): 10, (16, 16): 40, (16, 30): 99}


def windows():
    u = ctypes.windll.user32
    items = {}

    @ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    def visit(h, _):
        if not u.IsWindowVisible(h) or u.GetWindow(h, 4):
            return True
        pid = wintypes.DWORD()
        u.GetWindowThreadProcessId(h, ctypes.byref(pid))
        title = ctypes.create_unicode_buffer(1024)
        u.GetWindowTextW(h, title, len(title))
        if (
            title.value
            and pid.value != os.getpid()
            and title.value != "Program Manager"
        ):
            items[h] = title.value
        return True

    u.EnumWindows(visit, 0)
    return items


def activate(hwnd):
    u = ctypes.windll.user32
    if not u.IsWindow(hwnd):
        raise Stuck("所选窗口已经关闭，请刷新窗口列表。")
    if u.IsIconic(hwnd):
        u.ShowWindow(hwnd, 9)
    if u.GetForegroundWindow() == hwnd:
        return
    u.SetForegroundWindow(hwnd)
    # Cross-thread activation can return before the game owns the foreground.
    deadline = time.monotonic() + 1.0
    while u.GetForegroundWindow() != hwnd:
        if time.monotonic() >= deadline:
            raise Stuck("请将游戏切到前台，再点识别或开始。")
        time.sleep(0.02)


def game_window(page):
    """Bind the opened tab to its Chrome process; game state comes from screen pixels."""
    page.bring_to_front()
    session = page.context.browser.new_browser_cdp_session()
    try:
        processes = session.send("SystemInfo.getProcessInfo")["processInfo"]
        pid = next(p["id"] for p in processes if p["type"] == "browser")
    finally:
        session.detach()
    handles = {}
    user = ctypes.windll.user32
    for handle, title in windows().items():
        owner = wintypes.DWORD()
        user.GetWindowThreadProcessId(handle, ctypes.byref(owner))
        if owner.value == pid:
            handles[handle] = title
    if len(handles) == 1:
        return next(iter(handles))
    foreground = user.GetForegroundWindow()
    if foreground in handles and page.evaluate("document.hasFocus()"):
        return foreground
    title = page.title()
    matches = [h for h, t in handles.items() if title and t.startswith(title + " - ")]
    if len(matches) == 1:
        return matches[0]
    raise Stuck("网页已打开。请点选游戏窗口，以截图识别。")


def window_capture(hwnd, *, screen=False):
    area = wintypes.RECT()
    u = ctypes.windll.user32
    if not u.IsWindow(hwnd):
        raise Stuck("目标窗口已关闭。")
    if u.IsIconic(hwnd):
        u.ShowWindow(hwnd, 9)
    u.GetWindowRect(hwnd, ctypes.byref(area))
    rect = (area.left, area.top, area.right, area.bottom)
    title = ctypes.create_unicode_buffer(1024)
    u.GetWindowTextW(hwnd, title, len(title))
    # PrintWindow returns this HWND's client surface even if our own preview
    # covers it. Place it in the same physical coordinate system as live input.
    client = wintypes.RECT()
    origin = wintypes.POINT()
    u.GetClientRect(hwnd, ctypes.byref(client))
    u.ClientToScreen(hwnd, ctypes.byref(origin))
    try:
        if screen:
            raise OSError("capture the composited surface")
        surface = ImageGrab.grab(window=hwnd).convert("RGB")
        if (
            surface.size != (client.right, client.bottom)
            or max(b - a for a, b in surface.getextrema()) < 12
        ):
            raise OSError("no rendered client surface")
        image = Image.new("RGB", (rect[2] - rect[0], rect[3] - rect[1]), "#243033")
        image.paste(surface, (origin.x - rect[0], origin.y - rect[1]))
    except OSError:
        activate(hwnd)
        image = ImageGrab.grab(
            bbox=rect, all_screens=True, include_layered_windows=True
        ).convert("RGB")
    return rect, title.value, image


class Session:
    def __init__(self, commands, events, controls):
        self.commands, self.events, self.controls = commands, events, controls
        self.port = self.image = self.page = None

    def emit(self, event, **data):
        self.events.put((event, data))

    def record_recognition(self, stage, message):
        base = os.environ.get("LOCALAPPDATA")
        if not base:
            return
        record = dict(time=datetime.now().isoformat(timespec="seconds"), stage=stage,
                      shape=getattr(self, "shape", None), box=getattr(self, "box", None), detail=message)
        try:
            folder = Path(base) / "Mineswift" / "runs"
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / "recognition.jsonl").open("a", encoding="utf8") as file:
                print(json.dumps(record, ensure_ascii=False), file=file)
        except OSError:
            pass  # diagnostics must not prevent manual calibration

    def preview(self, frame):
        bounds = (
            self.port.screen_bounds(frame.layout)
            if hasattr(self.port, "screen_bounds")
            else frame.layout.bounds()
        )
        self.emit(
            "preview",
            board=visible_rows(frame.board) if frame.board is not None else [],
            status=frame.status,
            mines_left=frame.board.mines_left if frame.board is not None else None,
            bounds=tuple(map(round, bounds)),
            source=getattr(self.port, "name", "canvas"),
        )

    def bind(self, stack, spec):
        self.spec = spec
        self.port = self.image = self.page = None
        self.covered = self.box = self.detected_box = self.restart = None
        self.labels = {}
        self.shape = spec["shape"]
        self.hwnd = spec["hwnd"]
        if spec["url"]:
            from .web_browser import player_tab

            self.page = stack.enter_context(player_tab(spec["url"]))
            self.hwnd = game_window(self.page)
        if self.hwnd not in windows():
            raise Stuck("请从列表中选择已打开的游戏，或粘贴网址。")
        if "扫雷游戏网页版 - Minesweeper" in windows().get(self.hwnd, ""):
            try:
                activate(self.hwnd)
                port = ChromeGame(Sprites.load())
                port.hwnd = self.hwnd
                port.root = port.ui.ElementFromHandle(self.hwnd)
                port._bind_canvas()
                self.port = port
                self.preview(settled(port))
                return
            except (Stuck, Unreadable):
                self.port = None
        shapes = [self.shape] if self.shape else None
        self.window_rect, self.title, self.image = window_capture(self.hwnd)
        def candidates():
            pixels = np.asarray(self.image)
            matches = locate_lattices(pixels, shapes)
            if not matches and shapes is None:
                return locate_lattices(pixels, PRESETS)
            return matches

        matches = candidates()
        # Chrome can return a complete frame with its accelerated board layer
        # missing. Retry the visible surface while the page paints its tiles.
        deadline = time.monotonic() + 4
        while not matches and time.monotonic() < deadline:
            if self.controls.stop.wait(0.16):
                raise StopRequested("已取消识别。")
            self.window_rect, self.title, self.image = window_capture(
                self.hwnd, screen=True
            )
            matches = candidates()
        if len(matches) == 1:
            self.shape, self.box = matches[0]
            self.detected_box = self.box
            if "minesweeper online" in self.title.casefold():
                face = online_face(np.asarray(self.image), self.box, self.shape[1])
                if face:
                    self.restart = (
                        face[0] + self.window_rect[0],
                        face[1] + self.window_rect[1],
                    )
        self.emit("capture_image", image=self.image, box=self.box)
        if self.box is None:
            reason = (f"发现 {len(matches)} 个可能的棋盘" if matches else "画面中未找到边缘完整的规则棋盘")
            self.record_recognition("定位", reason)
            self.emit("hint", message=f"{reason}。请在预览上框出要玩的棋盘。")
            return
        self.calibrate({})

    def calibrate(self, changes):
        if self.image is None:
            raise Stuck("请先识别游戏窗口。")
        self.port = None
        if "box" in changes:
            self.box = changes["box"]
            self.covered = None
            self.labels = {}
            self.shape = self.spec["shape"]
        if not self.box:
            raise Stuck("请先框出棋盘。")
        left, top, right, bottom = self.box
        pixels = np.asarray(self.image.crop(self.box))
        shape = self.shape or (None, None)
        if self.box == self.detected_box and None not in shape:
            # The window-wide fit already verified all row/column centres.
            # Cropping away the outside edges can hide the grid on a nearly fresh board.
            rows, cols = shape
            cal = Calibration(rows, cols, 0, 0, (right-left)/cols, (bottom-top)/rows)
        else:
            cal = DesktopReader.fit(pixels, shape)
        self.shape = (cal.rows, cal.cols)
        if "restart" in changes:
            self.restart = (
                self.window_rect[0] + changes["restart"][0],
                self.window_rect[1] + changes["restart"][1],
            )
        if "point" in changes:
            x, y = changes["point"]
            cell = (
                int(np.floor((y - top - cal.y0) / cal.pitch_y)),
                int(np.floor((x - left - cal.x0) / cal.pitch_x)),
            )
            if not (0 <= cell[0] < cal.rows and 0 <= cell[1] < cal.cols):
                raise Stuck("请点在棋盘格的中央。")
            if changes["label"] == "covered":
                self.covered = cell
            else:
                self.labels[changes["label"]] = cell
        total = self.spec["mines"] or PRESETS.get(self.shape)
        if total is None:
            raise Stuck(f"已找到 {cal.rows} 行 × {cal.cols} 列。请在盘型中填写 "
                        f"{cal.rows}x{cal.cols}/总雷数，再点识别游戏。")
        from .desktop_grid import _tile

        reader = DesktopReader(
            pixels,
            self.shape,
            self.covered,
            total,
            rules=self.spec["rules"],
            classic_colors=self.spec["classic_colors"],
            calibration=cal,
        )
        for label, cell in self.labels.items():
            value = {"flag": -2, "hit": "lost", "mine": "lost"}.get(label)
            if value is None:
                value = int(label)
            reader.templates.setdefault(value, []).append(
                _tile(reader.crop(pixels, *cell))
            )
        box = (
            self.window_rect[0] + left,
            self.window_rect[1] + top,
            self.window_rect[0] + right,
            self.window_rect[1] + bottom,
        )
        port = DesktopGame.calibrated(
            self.hwnd, self.window_rect, self.title, box, reader, self.restart
        )
        if self.page is not None:
            port.page, port.page_url = self.page, self.page.url
        result = reader.decode(pixels, port.layout)
        self.port = port
        self.preview(result)

    def run(self):
        import comtypes

        comtypes.CoInitialize()
        try:
            with ExitStack() as stack:
                while True:
                    cmd, data = self.commands.get()
                    if cmd == "close":
                        return
                    try:
                        if cmd == "bind":
                            stack.close()
                            self.controls.stop.clear()
                            self.bind(stack, data)
                        elif cmd == "calibrate":
                            self.calibrate(data)
                        elif cmd == "start":
                            if self.port is None:
                                raise Stuck("请先完成识别。")
                            with ChromeGame.one_player():
                                if self.page:
                                    self.page.bring_to_front()
                                if self.hwnd:
                                    activate(self.hwnd)
                                self.port.controls = self.controls
                                self.controls.again.clear()
                                path = (
                                    Path(os.environ["LOCALAPPDATA"])
                                    / "Mineswift"
                                    / "runs"
                                )
                                path.mkdir(parents=True, exist_ok=True)
                                with (
                                    path
                                    / (
                                        datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                                        + ".jsonl"
                                    )
                                ).open("x", encoding="utf8") as log:

                                    def emit(name, **payload):
                                        print(
                                            json.dumps(
                                                {"event": name, **payload},
                                                ensure_ascii=False,
                                            ),
                                            file=log,
                                            flush=True,
                                        )
                                        self.emit(name, **payload)

                                    emit(
                                        "capture",
                                        source=getattr(self.port, "name", "canvas"),
                                        rules=self.spec["rules"],
                                    )
                                    try:
                                        frame = settled(self.port)
                                        run_rounds(
                                            self.port,
                                            frame,
                                            secrets.randbits(32),
                                            emit,
                                            controls=self.controls,
                                            hybrid=True,
                                            rules=self.spec["rules"],
                                        )
                                    except Exception as exc:
                                        if isinstance(exc, (ReadbackError, Unreadable)) and isinstance(self.port, DesktopGame):
                                            pixels = getattr(self.port, "last_pixels", None)
                                            if pixels is not None:
                                                crop = Image.fromarray(pixels)
                                                self.image = self.image.copy()
                                                self.image.paste(crop, self.box)
                                                self.covered, self.labels = None, {}
                                                self.emit("capture_image", image=self.image, box=self.box)
                                                screenshot = Path(log.name).with_suffix(".png")
                                                crop.save(screenshot)
                                                emit("readback_failed", board_image=screenshot.name)
                                        emit("control_returned" if isinstance(exc, StopRequested) else "stopped",
                                             reason=str(exc))
                                        raise
                    except StopRequested as exc:
                        self.emit("released", message=str(exc) or "鼠标已交还，可以继续自己玩。")
                    except Exception as exc:
                        if cmd == "bind" or cmd == "calibrate":
                            self.port = None
                        message = (
                            "棋盘上的数字与周围的格子矛盾，请核对最新预览并重新识别。"
                            if isinstance(exc, Inconsistent)
                            else str(exc) or type(exc).__name__
                        )
                        if cmd in ("bind", "calibrate"):
                            self.record_recognition("识别" if cmd == "bind" else "校准", message)
                        self.emit(
                            "error",
                            message=message,
                            ready=self.port is not None,
                        )
        finally:
            comtypes.CoUninitialize()
