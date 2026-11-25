"""Capture a bound Chrome document using its rendered grid or classic canvas."""

from contextlib import contextmanager
from pathlib import Path
import time
import os
import subprocess
import shutil
from urllib.parse import urlsplit

import numpy as np

from .board import Board, Stuck, Unreadable
from .web_chrome import BoardChanged, RestartRound, StopRequested, settled
from .web_vision import Frame, GridLayout

GRID = Path(__file__).with_name("web_grid.js").read_text(encoding="utf-8")
ROOTS = '#game:has(.square), #game:has(#CellsBlock .cell), [role="grid"]:has([role="gridcell"]), canvas#paf'
KEYS = """() => {
    if (!window.__msKeys) {
        window.__msKeys = {value:""};
        addEventListener("keydown", e => {
            if (e.repeat || e.ctrlKey || e.altKey || e.metaKey || e.shiftKey ||
                e.target.closest?.("input,textarea,[contenteditable=true]")) return;
            if (e.key === "Escape" || e.key.toLowerCase() === "r") window.__msKeys.value=e.key.toLowerCase();
        }, true);
    }
    const key = window.__msKeys.value; window.__msKeys.value="";
    return {visible: !document.hidden, key};
}"""


class BrowserBoard:
    def __init__(self, page, frame, root, mines=None):
        self.page, self.frame, self.root = page, frame, root
        self.mines = mines
        self.controls = None
        self.canvas = root.evaluate("e => e.matches('canvas#paf')")
        self.name = "canvas" if self.canvas else "dom"
        self.sprites = None
        self.top_url = page.url

    def guard(self):
        if (
            self.page.is_closed()
            or self.frame.is_detached()
            or self.page.url != self.top_url
        ):
            raise Stuck("绑定的标签或框架已经离开棋盘。")
        if not self.root.evaluate("e => e.isConnected"):
            raise Stuck("网页替换了棋盘，请重新连接。")
        states = [
            f.evaluate(KEYS) for f in dict.fromkeys((self.page.main_frame, self.frame))
        ]
        if any(not s["visible"] for s in states):
            raise Stuck("棋盘标签已经切到后台。")
        if any(s["key"] == "escape" for s in states) or (
            self.controls and self.controls.stop.is_set()
        ):
            raise StopRequested("已停止。")
        if self.controls and (
            any(s["key"] == "r" for s in states) or self.controls.again.is_set()
        ):
            self.controls.again.clear()
            raise RestartRound

    def _canvas_frame(self):
        import io
        from PIL import Image
        from .web_vision import Layout, Sprites, decode

        if self.frame != self.page.main_frame:
            raise Unreadable("画布模式需要顶层页面。")
        rects = self.root.evaluate("""grid => [grid, document.querySelector('#face'), document.querySelector('#rm')].map(e => {
            if (!e) throw Error('缺少画布指示器');
            const b=e.getBoundingClientRect();
            if (b.left<0 || b.top<0 || b.right>innerWidth || b.bottom>innerHeight ||
                document.elementFromPoint(b.x+b.width/2,b.y+b.height/2)!==e) throw Error('画布未完整显示');
            return [b.left,b.top,b.right,b.bottom].map(Math.round);
        })""")
        layout = Layout(*(tuple(b) for b in rects))
        if self.sprites is None:
            self.sprites = Sprites.load()
        with Image.open(io.BytesIO(self.page.screenshot(scale="css"))) as image:
            return decode(image.crop(layout.bounds()), layout, self.sprites)

    def read(self):
        from playwright.sync_api import Error

        self.guard()
        try:
            if self.canvas:
                return self._canvas_frame()
            data = self.root.evaluate(GRID, {"mines": self.mines})
        except Error as exc:
            raise Unreadable(str(exc).splitlines()[0]) from exc
        self.snapshot = data
        rows, cols = data["rows"], data["cols"]
        layout = GridLayout(tuple(data["grid"]), rows, cols)
        if data["status"] != "playing":
            return Frame(layout, data["status"])
        cells = np.array(data["values"], dtype=np.int8).reshape(rows, cols)
        return Frame(layout, "playing", Board(rows, cols, cells, data["remaining"]))

    def click(self, before, cell, button):
        if self.canvas:
            if not self.read().matches(before):
                raise BoardChanged("落子前棋盘发生变化。")
            x, y = before.layout.center(cell)
            # Use viewport coordinates for the canvas, including its CSS border.
            if (
                self.frame.evaluate(
                    "([x,y])=>document.elementFromPoint(x,y)?.id", [x, y]
                )
                != "paf"
            ):
                raise Stuck("画布落点被遮挡。")
            self.page.mouse.move(x, y)
        else:
            from playwright.sync_api import Error

            self.guard()
            if before.status != "playing" or before.board is None:
                raise Stuck("只能在进行中的棋盘落子。")
            board = before.board
            expected = dict(rows=board.rows, cols=board.cols, grid=before.layout.grid,
                            status=before.status, remaining=board.mines_left,
                            values=board.cells.ravel().tolist())
            idx = cell[0] * before.board.cols + cell[1]
            # The script reads every visible tile and hit-tests it, then compares
            # that snapshot and returns the target in the same browser call.
            try:
                target = self.root.evaluate_handle(GRID, {"mines": self.mines, "expected": expected, "target": idx})
            except Error as exc:
                if "决定落子后棋盘发生变化" in str(exc):
                    raise BoardChanged("落子前棋盘发生变化。") from exc
                raise Unreadable(str(exc).splitlines()[0]) from exc
            try:
                if button != "chord":
                    target.as_element().click(button=button, timeout=1200)
                    return
                target.as_element().hover(timeout=1200)
            finally:
                target.dispose()
        self.guard()
        if button != "chord":
            self.page.mouse.click(x, y, button=button)
            return
        mouse = self.page.mouse
        mouse.down(button="left")
        try:
            mouse.down(button="right")
            mouse.up(button="right")
        finally:
            mouse.up(button="left")

    def restart(self):
        previous = settled(self)
        online = not self.canvas and self.root.evaluate(
            "e => !!e.querySelector('#CellsBlock .cell')"
        )
        if previous.board is not None and previous.board.covered().all():
            return previous
        if self.canvas:
            self.frame.locator("#face").click(timeout=1200)
        else:
            face = self.root.evaluate_handle(
                GRID,
                {"expected": self.snapshot, "mines": self.mines, "target": "restart"},
            )
            try:
                if face.as_element() is None:
                    raise Stuck("此棋盘没有已识别的重开按钮。")
                face.as_element().click(timeout=1200)
            finally:
                face.dispose()
        if online:
            # A new round on minesweeper.online has a new permalink and can replace
            # the root element. Only this explicit face click may switch our binding.
            previous_url = urlsplit(self.top_url)
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                current_url = urlsplit(self.page.url)
                if (current_url.scheme, current_url.netloc) != (
                    previous_url.scheme,
                    previous_url.netloc,
                ):
                    raise Stuck("重开后页面离开了原站点。")
                roots = self.frame.locator(
                    "#game:visible:has(#CellsBlock .cell)"
                ).element_handles()
                if (
                    len(roots) == 1
                    and self.page.url != self.top_url
                    and "/game/" in current_url.path
                ):
                    self.root.dispose()
                    self.root, self.top_url = roots[0], self.page.url
                    break
                for root in roots:
                    root.dispose()
                time.sleep(0.08)
            else:
                raise Stuck("未确认同一标签已进入新的一局。")
        fresh = settled(self, before=previous)
        if (
            fresh.board is None
            or not fresh.board.covered().all()
            or fresh.layout.geometry()[:2] != previous.layout.geometry()[:2]
        ):
            raise Stuck("重开后未读到同尺寸新棋盘。")
        return fresh

    def screen_bounds(self, layout):
        # Only for positioning the optional Windows monitor. Input uses DOM coordinates.
        box = self.root.bounding_box()
        ox, oy, scale = self.page.evaluate(
            "()=>[screenX+(outerWidth-innerWidth)/2,screenY+outerHeight-innerHeight,devicePixelRatio]"
        )
        return tuple(
            round(v * scale)
            for v in (
                ox + box["x"],
                oy + box["y"],
                ox + box["x"] + box["width"],
                oy + box["y"] + box["height"],
            )
        )


def discover(pages, wait, mines):
    from playwright.sync_api import Error

    deadline = time.monotonic() + wait
    while True:
        found = []
        for page in pages():
            if not page.url.startswith(("http://", "https://")) or page.is_closed():
                continue
            if not page.evaluate("()=>!document.hidden"):
                continue
            for frame in page.frames:
                try:
                    for root in frame.locator(ROOTS).element_handles():
                        if root.is_visible():
                            found.append((page, frame, root))
                        else:
                            root.dispose()
                except Error:
                    if not frame.is_detached():
                        raise
        if len(found) == 1:
            return BrowserBoard(*found[0], mines)
        if found:
            raise Stuck("找到多个可见棋盘，请只保留要操作的棋盘。")
        if time.monotonic() >= deadline:
            raise Stuck("没有找到可识别的格子或画布。可改用 --desktop 框选棋盘。")
        time.sleep(0.2)


@contextmanager
def browser_session(url, cdp, wait, mines):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as api:
        if cdp:
            address = urlsplit(cdp)
            if address.scheme != "http" or address.hostname not in {
                "127.0.0.1",
                "localhost",
                "::1",
            }:
                raise ValueError("--cdp 需要本机 Chrome 的 HTTP 调试地址")
            browser = api.chromium.connect_over_cdp(cdp, timeout=5000)
            yield discover(
                lambda: [p for c in browser.contexts for p in c.pages], wait, mines
            )
        else:
            if urlsplit(url).scheme not in {"http", "https"}:
                raise ValueError("--url 需要 HTTP 或 HTTPS 网页地址")
            browser = api.chromium.launch(channel="chrome", headless=False)
            try:
                page = browser.new_page(no_viewport=True)
                page.goto(url, wait_until="domcontentloaded")
                yield discover(lambda: [page], wait, mines)
            finally:
                browser.close()


@contextmanager
def player_tab(url):
    """Keep the user's game open after the assistant releases the tab."""
    from playwright.sync_api import sync_playwright

    profile = Path(os.environ["LOCALAPPDATA"]) / "Mineswift" / "Chrome"
    profile.mkdir(parents=True, exist_ok=True)
    active = profile / "DevToolsActivePort"
    with sync_playwright() as api:

        def connect():
            try:
                port, endpoint = active.read_text().splitlines()[:2]
                if not port.isdigit() or not endpoint.startswith("/devtools/browser/"):
                    return None
                return api.chromium.connect_over_cdp(
                    f"ws://127.0.0.1:{int(port)}{endpoint}", timeout=1200
                )
            except (OSError, ValueError):
                return None
            except Exception as exc:
                from playwright.sync_api import Error

                if isinstance(exc, Error):
                    return None
                raise

        browser = connect()
        if browser is None:
            paths = [
                Path(os.getenv(k, "")) / "Google/Chrome/Application/chrome.exe"
                for k in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")
            ]
            executable = next(
                (str(p) for p in paths if p.is_file()), shutil.which("chrome")
            )
            if not executable:
                raise Stuck(
                    "未找到桌面 Chrome。请在自己的浏览器打开网页后，从窗口列表选择。"
                )
            subprocess.Popen(
                [
                    executable,
                    "--user-data-dir=" + str(profile),
                    "--remote-debugging-port=0",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "about:blank",
                ]
            )
            deadline = time.monotonic() + 15
            while browser is None and time.monotonic() < deadline:
                time.sleep(0.2)
                browser = connect()
            if browser is None:
                raise Stuck("未能连接游戏专用 Chrome。")
        context = browser.contexts[0]
        page = next((p for p in context.pages if p.url == url), None)
        if page is None:
            page = (
                next((p for p in context.pages if p.url == "about:blank"), None)
                or context.new_page()
            )
            page.goto(url, wait_until="domcontentloaded", timeout=25000)
        page.bring_to_front()
        yield page
