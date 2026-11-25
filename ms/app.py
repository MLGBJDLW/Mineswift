"""Single-window Mineswift player for existing games and web addresses."""

import ctypes
import queue
import re
import threading
import time
import tkinter as tk
from tkinter import ttk
from urllib.parse import urlsplit

from PIL import Image, ImageTk

from .app_session import Session, windows
from .play_web import Controls
from .web_monitor import (
    BG,
    FG,
    GOLD,
    MUTED,
    TEAL,
    ACTIONS,
    draw_board,
    monitor_workarea,
    placement,
    risk_text,
)

FORMATS = ("自动", "9 × 9 / 10", "16 × 16 / 40", "16 × 30 / 99")
AUTO_GAME = "自动查找已打开的游戏"


def game_titles(targets):
    return [
        title
        for title in targets
        if re.search(
            r"\bminesweeper(?:game)?\b|^mines(?:$|\s+[-\[])|扫雷|踩地雷|踩地雷遊戲", title, re.I
        )
    ]


MARKS = {
    "选择未开格": "covered",
    "框选棋盘": "box",
    "选择重开按钮": "restart",
    "标记旗子": "flag",
    "标记爆雷": "hit",
    "标记普通雷": "mine",
    **{f"标记数字 {n}": str(n) for n in range(9)},
}


class App:
    def __init__(self, root):
        self.root = root
        root.title("Mineswift")
        root.geometry("590x585")
        root.minsize(330, 340)
        root.configure(bg=BG)
        self.commands, self.events = queue.Queue(), queue.Queue()
        self.controls = Controls()
        self.thread = threading.Thread(
            target=Session(self.commands, self.events, self.controls).run, daemon=True
        )
        self.thread.start()
        self.ready = self.playing = self.busy = self.closing = False
        self.image = self.box = self.drag = self.bounds = self.last_spec = None
        self.original = False
        self.rows, self.target, self.neighbors, self.memory = [], None, [], set()
        self.waiting_for_click = None
        self.accent = TEAL
        self.header = tk.Frame(root, bg=BG)
        self.header.pack(fill="x", padx=16, pady=(12, 2))
        tk.Label(
            self.header, text="Mineswift", fg=FG, bg=BG, font=("Segoe UI", 17, "bold")
        ).pack(side="left")
        self.caption = tk.Label(
            self.header, text="就绪", fg=TEAL, bg=BG, font=("Segoe UI", 10)
        )
        self.caption.pack(side="right")
        self.form = tk.Frame(root, bg=BG)
        self.form.pack(fill="x", padx=16, pady=5)
        source_row = tk.Frame(self.form, bg=BG)
        source_row.pack(fill="x")
        self.target_box = ttk.Combobox(source_row)
        self.target_box.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(source_row, text="点选窗口", width=8, command=self.pick_window).pack(
            side="left"
        )
        self.identify = ttk.Button(
            source_row, text="识别游戏", width=9, command=self.bind
        )
        self.identify.pack(side="left", padx=(5, 0))
        prefs = tk.Frame(self.form, bg=BG)
        prefs.pack(fill="x", pady=(7, 0))
        tk.Label(prefs, text="盘型", bg=BG, fg=MUTED).pack(side="left")
        self.format = ttk.Combobox(prefs, values=FORMATS, width=14)
        self.format.set("自动")
        self.format.pack(side="left", padx=(5, 8))
        self.rule = ttk.Combobox(
            prefs,
            values=("经典八邻域", "上下左右四邻域"),
            width=14,
            state="readonly",
        )
        self.rule.current(0)
        self.rule.pack(side="left")
        self.colors = tk.BooleanVar(value=True)
        tk.Checkbutton(
            prefs,
            text="经典旗雷辅助",
            variable=self.colors,
            bg=BG,
            fg=FG,
            selectcolor=BG,
            activebackground=BG,
            activeforeground=FG,
        ).pack(side="right")
        self.message = tk.StringVar(
            value="打开一盘扫雷，点「识别游戏」。也可以粘贴网址。"
        )
        self.note = tk.Label(
            root,
            textvariable=self.message,
            bg=BG,
            fg=MUTED,
            anchor="w",
            justify="left",
            wraplength=540,
        )
        self.note.pack(fill="x", padx=16, pady=(2, 4))
        self.tools = tk.Frame(root, bg=BG)
        self.mark = ttk.Combobox(
            self.tools, values=list(MARKS), state="readonly", width=15
        )
        self.mark.current(0)
        self.mark.pack(side="left")
        self.mark.bind("<<ComboboxSelected>>", lambda _: self.view_original())
        ttk.Button(self.tools, text="原图 / 读盘", command=self.toggle_view).pack(
            side="left", padx=6
        )
        self.footer = tk.Frame(root, bg=BG)
        self.footer.pack(side="bottom", fill="x", padx=16, pady=(4, 12))
        self.start_btn = ttk.Button(self.footer, text="开始接管", command=self.start)
        self.start_btn.pack(side="right")
        self.restart_btn = ttk.Button(
            self.footer, text="重开", command=self.controls.again.set
        )
        self.restart_btn.pack(side="right", padx=5)
        self.auto = tk.BooleanVar(value=False)
        self.blind = tk.BooleanVar(value=False)
        for label, var, event in (
            ("自动续局", self.auto, self.controls.auto),
            ("盲扫", self.blind, self.controls.blind),
        ):
            tk.Checkbutton(
                self.footer,
                text=label,
                variable=var,
                bg=BG,
                fg=FG,
                selectcolor=BG,
                activebackground=BG,
                activeforeground=FG,
                command=lambda v=var, e=event: e.set() if v.get() else e.clear(),
            ).pack(side="left")
        self.stats = tk.StringVar()
        tk.Label(root, textvariable=self.stats, anchor="w", bg=BG, fg=FG).pack(
            side="bottom", fill="x", padx=16
        )
        self.canvas = tk.Canvas(root, bg="#203031", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=16, pady=4)
        self.canvas.bind("<Configure>", lambda _: self.draw())
        self.canvas.bind("<ButtonPress-1>", self.press)
        self.canvas.bind("<B1-Motion>", self.motion)
        self.canvas.bind("<ButtonRelease-1>", self.release)
        root.bind("<Escape>", lambda _: self.stop())
        root.bind(
            "<Key-r>", lambda e: self.controls.again.set() if self.playing else None
        )
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh()
        self.buttons()
        root.after(40, self.poll)

    def pick_window(self):
        if self.playing or self.busy:
            return
        from .executor import WindowsMouse, _window_at

        mouse = WindowsMouse()
        u = mouse.user32
        self.root.withdraw()
        overlay = tk.Toplevel(self.root)
        overlay.overrideredirect(True)
        overlay.configure(bg="black", cursor="crosshair")
        overlay.attributes("-alpha", 0.32)
        tk.Label(
            overlay,
            text="单击游戏窗口选择 · Esc 取消",
            bg="black",
            fg="white",
            font=("Segoe UI", 17, "bold"),
        ).pack(pady=24)
        overlay.attributes("-topmost", True)
        overlay.update_idletasks()
        hwnd = u.GetAncestor(overlay.winfo_id(), 2)
        u.SetWindowPos(
            hwnd,
            ctypes.c_void_p(-1),
            *(u.GetSystemMetrics(i) for i in (76, 77, 78, 79)),
            0x40,
        )
        overlay.focus_force()

        def finish(event, select):
            point = mouse.cursor()
            overlay.destroy()
            self.root.update_idletasks()
            chosen = _window_at(u, *point) if select else None
            self.root.deiconify()
            self.refresh()
            label = next((s for s, h in self.targets.items() if h == chosen), None)
            if label:
                self.target_box.set(label)
                self.bind()
            elif select:
                self.message.set("该位置没有游戏窗口，请再点选一次。")

        overlay.bind("<ButtonRelease-1>", lambda e: finish(e, True))
        overlay.bind("<Escape>", lambda e: finish(e, False))
        overlay.bind("<ButtonRelease-3>", lambda e: finish(e, False))
        self.message.set("单击游戏窗口即可选择，Esc 取消。")

    def refresh(self):
        old = self.target_box.get()
        entries = windows()
        self.targets = {}
        for h, title in entries.items():
            label = (
                title if list(entries.values()).count(title) == 1 else f"{title}  [{h}]"
            )
            self.targets[label] = h
        labels = sorted(
            self.targets,
            key=lambda s: (not any(k in s.lower() for k in ("mines", "扫雷")), s),
        )
        vals = (
            [AUTO_GAME]
            + labels
            + [
                "https://minesweeper.online/start/1",
                "https://minesweeperonline.com/",
                "https://www.minesweeper.cn/",
            ]
        )
        self.target_box["values"] = vals
        self.target_box.set(
            old if old in vals or old.startswith(("https://", "http://")) else AUTO_GAME
        )

    def spec(self):
        target = self.target_box.get().strip()
        url = target if urlsplit(target).scheme in ("http", "https") else None
        hwnd = self.targets.get(target)
        if not url and not hwnd:
            raise ValueError("请选择窗口或输入完整的扫雷网址。")
        fmt = self.format.get().strip()
        shape = mines = None
        if fmt != "自动":
            match = re.fullmatch(r"(\d+)\s*[xX×*]\s*(\d+)\s*/\s*(\d+)", fmt)
            if not match:
                raise ValueError("自定义盘型可填写为 12x18/35。")
            r, c, mines = map(int, match.groups())
            if not (3 <= r <= 60 and 3 <= c <= 60 and 0 < mines < r * c):
                raise ValueError("请核对行数、列数和总雷数。")
            shape = (r, c)
        return dict(
            hwnd=hwnd,
            url=url,
            shape=shape,
            mines=mines,
            rules=("classic", "orthogonal")[self.rule.current()],
            classic_colors=self.colors.get(),
        )

    def buttons(self):
        self.start_btn.configure(
            text="交还鼠标" if self.playing else "开始接管",
            state="normal"
            if self.playing or (self.ready and not self.busy)
            else "disabled",
        )
        self.identify.configure(
            state="disabled" if self.playing or self.busy else "normal"
        )
        self.restart_btn.configure(state="normal" if self.playing else "disabled")

    def bind(self):
        if self.busy or self.playing:
            return
        if self.target_box.get().strip() == AUTO_GAME:
            self.refresh()
            candidates = game_titles(self.targets)
            if len(candidates) != 1:
                self.pick_window()
                return
            self.target_box.set(candidates[0])
        try:
            spec = self.spec()
        except ValueError as exc:
            self.message.set(str(exc))
            return
        self.ready = False
        self.busy = True
        self.image = self.box = None
        self.rows = []
        self.original = False
        self.tools.pack_forget()
        self.stats.set("")
        self.last_spec = spec
        self.caption.config(text="正在识别")
        self.message.set("正在读取目标棋盘。")
        self.controls.stop.clear()
        self.buttons()
        self.draw()
        self.root.lower()
        self.commands.put(("bind", spec))

    def start(self):
        if self.playing:
            self.stop()
            return
        if not self.ready or self.busy:
            return
        try:
            if self.spec() != self.last_spec:
                self.message.set("目标或规则已修改，请重新识别。")
                return
        except ValueError as exc:
            self.message.set(str(exc))
            return
        self.playing = True
        self.original = False
        self.controls.stop.clear()
        self.form.pack_forget()
        self.tools.pack_forget()
        self.old_geometry = self.root.geometry()
        self.caption.config(text="接管中")
        self.message.set("Esc 或「交还鼠标」可以随时继续自己玩。")
        self.note.config(wraplength=285)
        hwnd = ctypes.windll.user32.GetAncestor(self.root.winfo_id(), 2)
        self.controls.panel_hwnd = hwnd
        style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
        ctypes.windll.user32.SetWindowLongW(hwnd, -20, style | 0x08000000)
        if self.bounds:
            room = placement(self.bounds, monitor_workarea(self.bounds), minimum=(330, 340))
            if room:
                w, h, x, y = room
                self.root.geometry(f"{max(w, 330)}x{max(h, 340)}")
                self.root.update_idletasks()
                ctypes.windll.user32.SetWindowPos(
                    hwnd, ctypes.c_void_p(-1), x, y, 0, 0, 0x11
                )
            else:
                self.root.iconify()
        self.buttons()
        self.commands.put(("start", {}))

    def stop(self):
        if self.playing:
            self.controls.stop.set()

    def idle(self):
        if self.playing:
            self.playing = False
            hwnd = self.controls.panel_hwnd
            style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
            ctypes.windll.user32.SetWindowLongW(hwnd, -20, style & ~0x08000000)
            self.controls.panel_hwnd = 0
            self.root.attributes("-topmost", False)
            if self.root.state() == "iconic":
                self.root.deiconify()
            self.root.geometry(self.old_geometry)
            self.form.pack(before=self.note, fill="x", padx=16, pady=5)
            self.note.config(wraplength=540)
        self.busy = False
        self.root.lift()
        if self.image is not None:
            self.tools.pack(before=self.canvas, fill="x", padx=16, pady=3)
        self.buttons()

    def view_original(self):
        self.original = True
        self.draw()

    def toggle_view(self):
        self.original = not self.original if self.ready else True
        self.draw()

    def draw(self):
        cv = self.canvas
        cv.delete("all")
        if self.original and self.image is not None:
            zoom = self.box and MARKS[self.mark.get()] not in ("box", "restart")
            view = self.image.crop(self.box) if zoom else self.image
            self.view_origin = self.box[:2] if zoom else (0, 0)
            w, h = view.size
            self.scale = min(
                max(1, cv.winfo_width()) / w, max(1, cv.winfo_height()) / h
            )
            size = (max(1, round(w * self.scale)), max(1, round(h * self.scale)))
            self.offset = (
                (cv.winfo_width() - size[0]) // 2,
                (cv.winfo_height() - size[1]) // 2,
            )
            self.photo = ImageTk.PhotoImage(
                view.resize(size, Image.Resampling.NEAREST if zoom else Image.Resampling.BILINEAR)
            )
            cv.create_image(*self.offset, image=self.photo, anchor="nw")
            if self.box and not zoom:
                x, y = self.offset
                cv.create_rectangle(
                    *(
                        v * self.scale + (x if i % 2 == 0 else y)
                        for i, v in enumerate(self.box)
                    ),
                    outline=TEAL,
                    width=2,
                )
        elif self.rows:
            draw_board(
                self.canvas,
                self.rows,
                self.target,
                self.neighbors,
                self.memory,
                self.accent,
            )
        else:
            cv.create_text(
                cv.winfo_width() / 2,
                cv.winfo_height() / 2,
                text="棋盘预览",
                fill=MUTED,
                font=("Segoe UI", 14),
            )

    def point(self, e):
        x, y = self.offset
        return (
            max(0, min(self.image.width, self.view_origin[0] + round((e.x - x) / self.scale))),
            max(0, min(self.image.height, self.view_origin[1] + round((e.y - y) / self.scale))),
        )

    def press(self, e):
        if self.original and self.image is not None and not self.busy:
            self.drag = self.point(e)

    def motion(self, e):
        if self.drag and self.mark.get() == "框选棋盘":
            x, y = self.drag
            r, b = self.point(e)
            self.box = (min(x, r), min(y, b), max(x, r), max(y, b))
            self.draw()

    def release(self, e):
        if self.drag is None:
            return
        start, self.drag = self.drag, None
        pos = self.point(e)
        mode = MARKS[self.mark.get()]
        if mode == "box":
            if abs(pos[0] - start[0]) < 12 or abs(pos[1] - start[1]) < 12:
                return
            self.box = (
                min(start[0], pos[0]),
                min(start[1], pos[1]),
                max(start[0], pos[0]),
                max(start[1], pos[1]),
            )
            changes = {"box": self.box}
            self.mark.current(0)
        elif mode == "restart":
            changes = {"restart": pos}
        else:
            changes = {"point": pos, "label": mode}
        self.busy = True
        self.ready = False
        self.buttons()
        self.commands.put(("calibrate", changes))

    def show(self, name, d):
        if name in ("observed", "decision", "resynced", "error", "released", "finished",
                    "input_retry", "flag_unavailable", "chord_unavailable"):
            self.waiting_for_click = None
        if name == "capture_image":
            self.image, self.box = d["image"], d["box"]
            self.original = True
            self.mark.current(0 if self.box else 1)
        elif name == "preview":
            self.rows = d["board"]
            self.bounds = d["bounds"]
            self.original = False
            self.ready = True
            self.target = None
            self.neighbors = []
            self.memory = set()
            self.caption.config(text="已识别")
            mode = {"desktop": "窗口截图", "dom": "网页格子", "canvas": "画布像素"}[
                d["source"]
            ]
            self.message.set(
                f"{mode} · {len(self.rows)} 行 × {len(self.rows[0])} 列 · 剩余 {d['mines_left']} 雷"
                if self.rows
                else "已识别终局"
            )
            self.idle()
        elif name in ("hint", "error", "released"):
            self.ready = d.get("ready", name == "released" and self.ready)
            if name == "error" and self.image is not None:
                self.original = True
            terminal = self.caption.cget("text") in ("胜利", "踩雷")
            if name != "released" or not terminal:
                self.caption.config(text={"hint": "校准", "error": "需要处理", "released": "已交还"}[name])
            self.message.set(d["message"])
            self.idle()
        elif name == "attached":
            self.caption.config(text="已接管")
            self.message.set("棋盘已连接，正在准备求解器和模型。")
            self.rows = d.get("board", [])
            self.target = None
            self.neighbors = []
            self.memory = set()
        elif name == "decision":
            self.rows = d["board"]
            self.target = (d["row"], d["col"])
            self.neighbors = list(map(tuple, d["targets"]))
            self.memory = set(map(tuple, d["memory"]))
            self.accent = GOLD if d["source"] == "model" else TEAL
            self.caption.config(text=ACTIONS[d["action"]])
            risk = f"  含雷 {risk_text(d['mine_risk'])}" if "mine_risk" in d else ""
            self.message.set(
                f"第 {d['row']} 行，第 {d['col']} 列{risk}。{d['explanation']}"
            )
        elif name == "input_sent":
            self.waiting_for_click = (time.monotonic(), d)
        elif name == "flag_unavailable":
            self.message.set("右键后整盘持续无变化，本局改为记住已证实的雷，用左键继续。")
        elif name == "chord_unavailable":
            self.message.set("双键后棋盘持续无变化，这一局改用左键逐格展开。")
        elif name == "input_retry":
            self.message.set(f"第 {d['row']} 行，第 {d['col']} 列左击后无变化，正在复核并重试一次。")
        elif name == "resynced":
            self.message.set("棋盘继续展开，已重新读盘并计算下一步。")
        elif name == "observed" and d.get("board"):
            self.rows = d["board"]
        elif name in ("finished", "waiting"):
            self.caption.config(text="胜利" if d["status"] == "won" else "踩雷")
            self.message.set("可重开，或交还鼠标继续自己玩。")
        if "clicks" in d:
            n = d["clicks"]
            self.stats.set(
                f"确定步 {n.get('solver', 0)}    双键 {n.get('chords', 0)}    模型猜格 {n.get('model', 0)}"
            )

    def poll(self):
        if self.closing and not self.thread.is_alive():
            self.root.destroy()
            return
        self.root.after(45, self.poll)
        redraw = False
        while not self.events.empty():
            name, data = self.events.get_nowait()
            if not self.closing:
                self.show(name, data)
                redraw |= name in ("capture_image", "preview", "hint", "error", "released",
                                   "attached", "decision", "observed", "finished", "waiting")
        if self.waiting_for_click and time.monotonic() - self.waiting_for_click[0] >= .25:
            _, click = self.waiting_for_click
            key = {"left": "左键", "right": "右键", "chord": "双键"}[click["button"]]
            self.message.set(f"第 {click['row']} 行第 {click['col']} 列{key}已发出，正在等棋盘反馈。")
            self.waiting_for_click = None
        if redraw:
            self.draw()

    def close(self):
        self.closing = True
        self.controls.stop.set()
        self.commands.put(("close", {}))
        self.root.withdraw()


def main(target=None):
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
    root = tk.Tk()
    app = App(root)
    if target and (
        target in app.targets or urlsplit(target).scheme in ("http", "https")
    ):
        app.target_box.set(target)
        root.after(200, app.bind)
    root.mainloop()


if __name__ == "__main__":
    main()
