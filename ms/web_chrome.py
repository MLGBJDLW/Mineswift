"""Locate the existing Chrome game using accessibility, then capture and click it."""
from __future__ import annotations

import ctypes
import time
from contextlib import contextmanager
from urllib.parse import urlsplit

from PIL import ImageGrab

from .board import Stuck, Unreadable
from .executor import WindowsMouse, _window_at
from .web_vision import Layout, decode


class RestartRound(Exception):
    pass


class StopRequested(Stuck):
    pass


class ReadbackError(Stuck):
    """The captured board did not reach a readable, stable state."""


class ChromeGame:
    def __init__(self, sprites):
        self.mouse = WindowsMouse()
        from comtypes.client import CreateObject, GetModule
        self.u = GetModule("UIAutomationCore.dll")
        self.ui = CreateObject(self.u.CUIAutomation, interface=self.u.IUIAutomation)
        self.sprites = sprites
        self.hwnd = 0
        self.controls = None
        self.restarting = False
        self.r_down = False
        self.mouse.user32.GetAsyncKeyState(0x1B)  # discard Escape presses before startup
        self.mouse.user32.GetAsyncKeyState(ord("R"))

    @staticmethod
    @contextmanager
    def one_player():
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateMutexW.restype = ctypes.c_void_p
        handle = kernel.CreateMutexW(None, False, "Local\\MineswiftWebPlayer")
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        exists = ctypes.get_last_error() == 183
        try:
            if exists:
                raise Stuck("已有一个 Mineswift 进程在接管棋盘。")
            yield
        finally:
            kernel.CloseHandle(ctypes.c_void_p(handle))

    def _find(self, parent, prop, value):
        return parent.FindFirst(4, self.ui.CreatePropertyCondition(prop, value))

    def _all(self, parent, scope, prop, value):
        items = parent.FindAll(scope, self.ui.CreatePropertyCondition(prop, value))
        return [items.GetElement(i) for i in range(items.Length)]

    def _host_ok(self):
        edit = self._find(self.root, self.u.UIA_AutomationIdPropertyId, "view_1012")
        if not edit:
            return False
        url = str(edit.GetCurrentPropertyValue(self.u.UIA_ValueValuePropertyId))
        return urlsplit(url if "://" in url else "//" + url).hostname in {"minesweeper.cn", "www.minesweeper.cn"}

    def _windows(self):
        # Filter top-level HWNDs before asking accessibility providers for descendants.
        # Other Chromium applications can have slow or suspended UIA providers.
        user32 = self.mouse.user32
        handles = []

        @ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
        def visit(hwnd, _):
            title = ctypes.create_unicode_buffer(1024)
            user32.GetWindowTextW(hwnd, title, len(title))
            if title.value.endswith(" - Google Chrome") and user32.IsWindowVisible(hwnd):
                cls = ctypes.create_unicode_buffer(128)
                user32.GetClassNameW(hwnd, cls, len(cls))
                if cls.value == "Chrome_WidgetWin_1":
                    handles.append(hwnd)
            return True

        user32.EnumWindows(visit, 0)
        return [self.ui.ElementFromHandle(hwnd) for hwnd in handles]

    def attach(self, wait):
        deadline = time.monotonic() + wait
        while True:
            self.check_escape()
            matches = []
            for root in self._windows():
                strip = self._find(root, self.u.UIA_AutomationIdPropertyId, "view_21")
                for tab in self._all(strip or root, 4, self.u.UIA_ControlTypePropertyId, self.u.UIA_TabItemControlTypeId):
                    if "扫雷游戏网页版 - Minesweeper" in tab.CurrentName:
                        pattern = tab.GetCurrentPattern(self.u.UIA_SelectionItemPatternId)
                        selection = pattern.QueryInterface(self.u.IUIAutomationSelectionItemPattern)
                        matches.append((root, selection, bool(selection.CurrentIsSelected), tab))
            if matches:
                break
            if time.monotonic() >= deadline:
                raise Stuck("没找到已打开的 minesweeper.cn 棋盘。打开或恢复原标签后，可再次启动。")
            time.sleep(.45)
        if len(matches) > 1:
            matches = [m for m in matches if m[2] and m[0].CurrentNativeWindowHandle == self.mouse.foreground_window()]
        if len(matches) != 1:
            raise Stuck("有多盘扫雷，请把要接着玩的那盘切到前台。")
        self.root, selection, selected, tab = matches[0]
        self.hwnd = self.root.CurrentNativeWindowHandle
        user32 = self.mouse.user32
        if user32.IsIconic(self.hwnd):
            user32.ShowWindow(self.hwnd, 9)
        if not selected:
            selection.Select()
        if self.mouse.foreground_window() != self.hwnd and not user32.SetForegroundWindow(self.hwnd):
            from comtypes import COMError
            try:
                tab.SetFocus()
            except COMError:
                pass
        self._bind_canvas()

    def _bind_canvas(self, old=None):
        deadline = time.monotonic() + (8 if old is not None else 4)
        while time.monotonic() < deadline:
            self.check_escape()
            if self.mouse.foreground_window() == self.hwnd and self._host_ok():
                doc = self._find(self.root, self.u.UIA_ControlTypePropertyId, self.u.UIA_DocumentControlTypeId)
                if doc:
                    nodes = [self._find(doc, self.u.UIA_AutomationIdPropertyId, name) for name in ("paf", "face", "rm")]
                    if all(nodes) and (old is None or tuple(nodes[0].GetRuntimeId()) != old):
                        self.nodes = nodes
                        return
            time.sleep(.1)
        if self.mouse.foreground_window() != self.hwnd:
            raise Stuck("Chrome 未取得前台。请切到棋盘再启动。")
        raise Stuck("Chrome 未显示可识别的目标画布。请让原棋盘在一个可见的 Chrome 窗口中打开。")

    def check_escape(self):
        if self.controls and self.controls.stop.is_set():
            raise StopRequested("观察窗已停止接管。")
        if self.mouse.user32.GetAsyncKeyState(0x1B) & 0x8001:
            if self.controls:
                self.controls.stop.set()
            raise StopRequested("已按 Esc，停止接管。")

    def guard(self):
        self.check_escape()
        foreground = self.mouse.foreground_window()
        if self.controls and foreground and foreground == self.controls.panel_hwnd:
            self.mouse.user32.SetForegroundWindow(self.hwnd)
            foreground = self.mouse.foreground_window()
        if foreground != self.hwnd:
            raise Stuck("前台窗口已切换，已归还鼠标。")
        if not self._host_ok():
            raise Stuck("当前标签已离开扫雷网页，已归还鼠标。")
        if self.controls and not self.restarting:
            keys = self.mouse.user32.GetAsyncKeyState
            state = keys(ord("R"))
            pressed = bool(state & 0x8001) and not self.r_down
            self.r_down = bool(state & 0x8000)
            if self.controls.again.is_set():
                self.controls.again.clear()
                raise RestartRound
            if pressed and not any(keys(k) & 0x8000 for k in (0x10, 0x11, 0x12, 0x5B, 0x5C)):
                focused = self.ui.GetFocusedElement()
                if focused and focused.CurrentControlType != self.u.UIA_EditControlTypeId:
                    raise RestartRound
        while any(self.mouse.user32.GetAsyncKeyState(key) & 0x8000 for key in (1, 2, 4)):
            cursor = self.mouse.cursor()
            if (not self.controls or not self.controls.panel_hwnd or not cursor
                    or _window_at(self.mouse.user32, *cursor) != self.controls.panel_hwnd):
                raise Stuck("检测到鼠标按键已按下，已归还鼠标。")
            self.check_escape()
            time.sleep(.02)

    def layout(self):
        self.guard()
        boxes = []
        for node in self.nodes:
            rect = node.CurrentBoundingRectangle
            if node.CurrentIsOffscreen or rect.right <= rect.left or rect.bottom <= rect.top:
                raise Stuck("当前窗口没有完整显示棋盘和计数器。")
            boxes.append((rect.left, rect.top, rect.right, rect.bottom))
        return Layout(*boxes)

    def read(self):
        layout = self.layout()
        image = ImageGrab.grab(bbox=layout.bounds(), all_screens=True)
        return decode(image, layout, self.sprites)

    def _click_node(self, frame, point, button, node):
        if not self.read().matches(frame):
            raise Stuck("决定落子后棋盘发生了变化。")
        x, y = point
        if _window_at(self.mouse.user32, x, y) != self.hwnd:
            raise Stuck("落点被其他窗口遮挡。")
        pt = self.u.tagPOINT(x, y)
        hit = self.ui.ElementFromPoint(pt)
        if not self.ui.CompareElements(hit, node):
            raise Stuck("落点处的网页元素已改变。")
        self.mouse.move(x, y)
        self.guard()
        if self.mouse.cursor() != (x, y) or self.layout() != frame.layout:
            raise Stuck("落点坐标改变。")
        with self.mouse.grabbed():
            if button == "chord":
                self.mouse.pressed_together(("left", "right"))
                self.mouse.released_together(("left", "right"))
            else:
                self.mouse.press(button)
                self.mouse.release(button)

    def click(self, frame, cell, button):
        self._click_node(frame, frame.layout.center(cell), button, self.nodes[0])

    def restart(self):
        self.restarting = True
        try:
            frame = settled(self)
            if frame.board is not None and frame.board.covered().all():
                return frame
            if frame.status == "playing":
                # An explicit mid-game restart reloads this verified tab. The site
                # restores the chosen difficulty and gives the new canvas a new ID.
                self.guard()
                reload_button = self._find(self.root, self.u.UIA_AutomationIdPropertyId, "view_1003")
                if not reload_button or reload_button.CurrentName not in ("重新加载", "Reload"):
                    raise Stuck("未找到当前 Chrome 标签的重新加载按钮。")
                old = tuple(self.nodes[0].GetRuntimeId())
                reload_button.GetCurrentPattern(self.u.UIA_InvokePatternId).QueryInterface(
                    self.u.IUIAutomationInvokePattern).Invoke()
                self._bind_canvas(old)
            else:
                x, y, r, b = frame.layout.face
                self._click_node(frame, ((x+r)//2, (y+b)//2), "left", self.nodes[1])
            fresh = settled(self, before=frame)
            if (fresh.status != "playing" or not fresh.board.covered().all()
                    or fresh.layout.geometry()[:2] != frame.layout.geometry()[:2]):
                raise Stuck("重开后没有读到同尺寸的全新棋盘。")
            return fresh
        finally:
            self.restarting = False
            if self.controls:
                self.controls.again.clear()
                self.r_down = bool(self.mouse.user32.GetAsyncKeyState(ord("R")) & 0x8000)


def settled(port, before=None, timeout=3, *, allow_unchanged=False):
    """Require two matching frames, and after an input require a visible change."""
    deadline = time.monotonic() + timeout
    last = None
    unchanged = False
    clean = True
    detail = "连续两帧未能确认同一局面"
    while time.monotonic() < deadline:
        try:
            current = port.read()
            if before is not None and not current.matches(before):
                clean = False
            unchanged = False
            if last is not None and current.matches(last):
                if before is None or not current.matches(before):
                    return current
                unchanged = True
            last = current
        except Unreadable as exc:
            clean = False
            detail, last, unchanged = str(exc), None, False
        # Look for the next rendered frame promptly. Keep the longer polling
        # interval for repeated stale frames and unreadable animations.
        time.sleep(.045 if unchanged or last is None else .018)
    if not clean and hasattr(port, "covered_terminal"):
        terminal = port.covered_terminal()
        if terminal is not None:
            return terminal
    if unchanged and clean:
        if allow_unchanged:
            return last
        raise Stuck("输入后棋盘持续无变化。")
    raise ReadbackError(f"未获得稳定读盘：{detail}。")
