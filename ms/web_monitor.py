"""Passive Tk decision monitor. Only the player thread owns Chrome and its mouse."""
import ctypes
from ctypes import wintypes
from queue import Empty, Queue
from threading import Thread
import tkinter as tk


ACTIONS = {"opening": "随机首开", "open": "求解器 · 安全格", "flag": "求解器 · 确定雷",
           "unflag": "撤去旧旗", "chord": "求解器 · 双键开格", "guess": "模型 · 必须猜测"}
BG, FG, MUTED, TEAL, GOLD = "#172325", "#eef6f3", "#a6bbb5", "#83d9b5", "#ffc774"


def risk_text(p):
    if p < .001:
        return "<0.1%"
    if p > .999:
        return ">99.9%"
    return f"{p:.1%}"


def placement(bounds, work, minimum=(218, 195)):
    """Fit beside the entire capture area (including native window decorations)."""
    x, y, r, b = bounds
    l, t, right, bottom = work
    for left, top, edge, base in ((r+12,t,right,bottom), (l,t,x-12,bottom),
                                  (l,b+12,right,bottom), (l,t,right,y-12)):
        width, height = min(350, edge-left-20), min(418, base-top-52)
        if width >= minimum[0] and height >= minimum[1]:
            return width, height, max(left,min(right-width-20, x)), max(top,min(base-height-52, y))
    return None


def monitor_workarea(bounds):
    class Info(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("screen", wintypes.RECT),
                    ("work", wintypes.RECT), ("flags", wintypes.DWORD)]
    user = ctypes.windll.user32
    user.MonitorFromRect.restype = ctypes.c_void_p
    rect = wintypes.RECT(*map(round,bounds))
    handle = user.MonitorFromRect(ctypes.byref(rect), 2)
    info = Info()
    info.size = ctypes.sizeof(info)
    if not user.GetMonitorInfoW(ctypes.c_void_p(handle), ctypes.byref(info)):
        raise ctypes.WinError()
    return info.work.left, info.work.top, info.work.right, info.work.bottom


def draw_board(cv, rows, target, neighbors, memory, accent):
    cv.delete("all")
    if not rows:
        return
    row_count, cols = len(rows), len(rows[0])
    width, height = cv.winfo_width(), cv.winfo_height()
    pitch = min(24, width//cols, height//row_count)
    if pitch < 2:
        return
    ox, oy = (width-pitch*cols)//2, (height-pitch*row_count)//2
    for r, row in enumerate(rows, 1):
        for c, value in enumerate(row, 1):
            pos = (r,c)
            x, y = ox+(c-1)*pitch, oy+(r-1)*pitch
            color = "#e2ede7" if value.isdigit() else "#415653"
            if pos in neighbors:
                color = "#a6d7bc"
            if pos == target:
                color = accent
            cv.create_rectangle(x, y, x+pitch-1, y+pitch-1, fill=color, width=0)
            if pos == target and pitch >= 8:
                cv.create_rectangle(x+1, y+1, x+pitch-2, y+pitch-2, outline="#133e3e", width=2)
            if pos in memory and value == "#":
                cv.create_oval(x+pitch*.28,y+pitch*.28,x+pitch*.72,y+pitch*.72,outline=GOLD,width=2)
            if value not in ("#", "0") and pitch >= 8:
                ink = {"F":"#cf4336", "1":"#18509e", "2":"#146347", "3":"#9c3833"}.get(value, "#333d59")
                cv.create_text(x+pitch/2, y+pitch/2, text="⚑" if value=="F" else value,
                               fill=ink, font=("Segoe UI", -max(8,int(pitch*.70)), "bold"))
    if target:
        r,c = target
        x,y = ox+(c-.5)*pitch, oy+(r-.5)*pitch
        cv.create_line(x, max(0,oy-6), x, oy+2, fill=accent, width=3)
        cv.create_line(max(0,ox-6), y, ox+2, y, fill=accent, width=3)


class Monitor:
    def __init__(self, controls):
        self.controls = controls
        self.root = tk.Tk()
        root = self.root
        root.withdraw()
        root.title("Mineswift · 决策观察")
        root.configure(bg=BG)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.rows, self.target, self.neighbors = [], None, []
        self.last_guess = None
        self.memory = set()
        self.blind = False
        self.accent = TEAL
        self.status = self.label("正在寻找棋盘", 19, FG)
        self.detail = self.label("", 13, MUTED)
        self.reason = self.label("", 13, FG)
        self.reason.config(wraplength=310, justify="left")
        footer = tk.Frame(root, bg=BG)
        footer.pack(side="bottom", fill="x", padx=10, pady=(2, 8))
        self.repeat = tk.BooleanVar(value=False)
        self.auto_btn = tk.Checkbutton(footer, text="自动续局", font=("Segoe UI", -13), bg=BG, fg=FG,
                                      selectcolor=BG, activebackground=BG, activeforeground=FG,
                                      variable=self.repeat, command=self.toggle_auto)
        self.auto_btn.pack(side="left")
        self.blind_var = tk.BooleanVar(value=controls.blind.is_set())
        tk.Checkbutton(footer, text="下局盲扫", font=("Segoe UI", -13), bg=BG, fg=FG, selectcolor=BG,
                       activebackground=BG, activeforeground=FG, variable=self.blind_var,
                       command=self.toggle_blind).pack(side="left")
        button_style = dict(font=("Segoe UI", -13), relief="flat", bd=0, padx=8, pady=4, takefocus=0)
        self.restart_btn = tk.Button(footer, text="重开 R", command=controls.again.set,
                                     bg=TEAL, fg="#153b32", **button_style)
        self.restart_btn.pack(side="right")
        self.stop_btn = tk.Button(footer, text="停止", command=self.close,
                                  bg="#304140", fg=FG, **button_style)
        self.stop_btn.pack(side="right", padx=4)
        self.stats = self.label("", 12, MUTED, side="bottom")
        self.guess_note = tk.Label(root, text="", bg=BG, fg=GOLD, anchor="w", font=("Segoe UI", -13))
        self.canvas = tk.Canvas(root, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=4)
        self.canvas.bind("<Configure>", lambda _: self.draw())
        # Map Tk before the worker activates Chrome. Mapping a withdrawn Tk window
        # later may activate it even when the following SetWindowPos uses NOACTIVATE.
        root.geometry("340x370+0+0")
        root.attributes("-alpha", 0.0)
        root.deiconify()
        root.update()
        user = ctypes.windll.user32
        hwnd = user.GetAncestor(root.winfo_id(), 2)
        style = user.GetWindowLongW(hwnd, -20)
        user.SetWindowLongW(hwnd, -20, style | 0x08000000 | 0x00000080)  # NOACTIVATE | TOOLWINDOW
        self.hwnd, self.bounds = hwnd, None
        controls.panel_hwnd = hwnd
        self.done = False

    def label(self, value, size, color, side="top"):
        widget = tk.Label(self.root, text=value, bg=BG, fg=color, anchor="w", font=("Segoe UI", -size))
        widget.pack(side=side, fill="x", padx=12, pady=(4,0))
        return widget

    def toggle_auto(self):
        if self.repeat.get():
            self.controls.auto.set()
        else:
            self.controls.auto.clear()

    def toggle_blind(self):
        if self.blind_var.get():
            self.controls.blind.set()
        else:
            self.controls.blind.clear()

    def close(self):
        self.controls.stop.set()
        self.root.withdraw()
        if self.done:
            self.root.quit()

    def dock(self, bounds):
        if self.bounds == bounds:
            return
        self.bounds = bounds
        room = placement(bounds, monitor_workarea(bounds))
        if not room:
            ctypes.windll.user32.ShowWindow(self.hwnd, 0)
            return
        w,h,x,y = room
        self.root.geometry(f"{w}x{h}")
        self.root.update_idletasks()
        # Place the real decorated HWND in physical screen coordinates, including
        # monitors with negative origins. Showing the monitor must not focus it.
        user = ctypes.windll.user32
        self.root.attributes("-alpha", 1.0)
        self.root.update_idletasks()
        hwnd = user.GetAncestor(self.root.winfo_id(), 2)
        self.controls.panel_hwnd = hwnd
        if not user.SetWindowPos(hwnd, ctypes.c_void_p(-1), x, y, 0, 0, 0x0051):
            raise ctypes.WinError()
        self.reason.config(wraplength=w-24)

    def draw(self):
        draw_board(self.canvas, self.rows, self.target, self.neighbors, self.memory, self.accent)

    def show(self, event, data):
        if event == "layout":
            self.dock(tuple(data["bounds"]))
        if data.get("board") is not None:
            self.rows = data["board"]
        if event == "attached":
            self.target, self.neighbors = None, []
            self.last_guess = None
            self.memory = set()
            self.blind = data.get("blind", False)
            self.guess_note.pack_forget()
            self.status.config(text=f"第 {data['round']} 局", fg=FG)
            self.detail.config(text="读取当前可见棋盘")
            self.reason.config(text="")
            self.stats.config(text="")
        elif event == "decision":
            action = data["action"]
            self.accent = GOLD if action == "guess" else TEAL
            self.target = (data["row"], data["col"])
            self.neighbors = list(map(tuple, data["targets"]))
            self.memory = set(map(tuple, data["memory"]))
            self.status.config(text=ACTIONS[action], fg=self.accent)
            line = f"第 {self.target[0]} 行 · 第 {self.target[1]} 列"
            if self.neighbors:
                line += f"   展开 {len(self.neighbors)} 格"
            if action == "guess" and "mine_risk" in data:
                self.last_guess = (self.target, risk_text(data["mine_risk"]))
                self.guess_note.config(text=f"本格含雷 {self.last_guess[1]}    全盘最低 {risk_text(data['min_risk'])}")
                self.guess_note.pack(before=self.canvas, fill="x", padx=12)
            self.detail.config(text=line)
            self.reason.config(text=data["explanation"])
        elif event == "observed" and self.last_guess and self.last_guess[0] == self.target:
            (r,c), risk = self.last_guess
            outcome = "踩雷" if data["status"] == "lost" else "安全"
            self.guess_note.config(text=f"最近猜测  ({r}, {c})   含雷 {risk}   ·   实际{outcome}")
        elif event in ("finished", "waiting"):
            self.status.config(text="胜利" if data["status"] == "won" else "踩雷", fg=FG)
            self.detail.config(text="R 重开  ·  Esc 停止")
            self.reason.config(text="棋盘保留最后一次决策" if self.rows else "")
            if not data.get("board") and event == "waiting" and not self.rows:
                self.canvas.delete("all")
        elif event == "stopped":
            self.status.config(text="已停止", fg=GOLD)
            self.detail.config(text=data["reason"], wraplength=295)
            if self.bounds is None and not self.controls.stop.is_set():
                self.root.geometry("320x225")
                self.root.attributes("-alpha", 1.0)
        if "clicks" in data:
            n = data["clicks"]
            mode = f"盲扫 · 记雷 {len(self.memory)}" if self.blind else f"双键 {n.get('chords',0)}"
            self.stats.config(text=f"已执行  ·  确定步 {n.get('solver',0)}    {mode}    猜测 {n.get('model',0)}")
        self.draw()


def watch(work, controls, finite):
    # Tk and the UIA worker must agree about physical pixels before creating HWNDs.
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
    panel, messages = Monitor(controls), Queue()
    if finite:
        panel.repeat.set(True)
        panel.auto_btn.config(state="disabled")
    result = []

    def worker():
        import comtypes
        comtypes.CoInitialize()
        try:
            result.append(work(lambda event, data: messages.put((event, data))))
        except Exception as exc:
            messages.put(("stopped", {"reason":str(exc)}))
            result.append(2)
        finally:
            comtypes.CoUninitialize()

    thread = Thread(target=worker, name="Mineswift player")
    thread.start()

    def poll():
        try:
            while True:
                panel.show(*messages.get_nowait())
        except Empty:
            pass
        if not thread.is_alive():
            panel.done = True
            panel.restart_btn.config(state="disabled")
            panel.stop_btn.config(text="关闭")
            if controls.stop.is_set() or finite:
                panel.root.quit()
                return
        panel.root.after(35, poll)

    panel.root.after(0, poll)
    try:
        panel.root.mainloop()
    finally:
        controls.stop.set()
        thread.join()
        panel.root.destroy()
    return result[0] if result else 2
