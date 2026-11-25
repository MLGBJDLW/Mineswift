"""Executor: what it does at a site, on a real board, with simulated timing.

The unit tests (``tests/test_executor.py``) pin each mechanism in isolation. This probe
answers the questions a user actually asks before trusting the executor with the mouse:

  1. does the settle loop wait for the picture to stop moving, and refuse a board that
     never stops?
  2. after a click that did not land, does it say so instead of clicking again?
  3. on a live board read by ``ms.perception``, how many mouse actions does the chord
     planner save, and does it verify each one?
  4. which chord form does a site model accept - middle, left+right-together, or neither -
     and does the probe find out by looking at the screen rather than at its own intent?
  5. (with --hwnd) does the Windows ``SendInput`` backend work here: DPI, cursor read-back,
     press/release through one batch?

Run:
    python experiments/executor_probe.py                 # headless, no real input
    python experiments/executor_probe.py --hwnd 123456   # plus the real mouse backend

Exit code 0 means every headless check passed. The --hwnd section only reports what it
observed; it never clicks anything.
"""
from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ms.executor as ex                                        # noqa: E402
from ms.board import COVERED, FLAG, Board                       # noqa: E402
from ms.chording import chord_candidates, plan_actions          # noqa: E402
from ms.engine import EXPERT, Minefield                         # noqa: E402
from ms.perception import Perception, Rect                      # noqa: E402

from tests._synthetic import Theme, render                      # noqa: E402

THEME = Theme(
    cell=30,
    covered=(176, 178, 184), covered_light=(214, 216, 222), covered_dark=(132, 134, 140),
    bg=(198, 200, 206), ink=(30, 34, 44), grid=(96, 98, 104),
    flag=(188, 40, 44), margin=12,
)


class Clock:
    """time.perf_counter/time.sleep under test control; sleeps are instant."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []
        self._real = (time.perf_counter, time.sleep)

    def __enter__(self) -> "Clock":
        time.perf_counter = lambda: self.now
        time.sleep = self._sleep
        return self

    def _sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.now += max(0.0, float(s))

    def __exit__(self, *exc) -> None:
        time.perf_counter, time.sleep = self._real


def render_board(cells: np.ndarray) -> np.ndarray:
    img, _ = render(np.asarray(cells, dtype=np.int8), THEME)
    return img


def board_of(cells: np.ndarray, mines_left: int = 0) -> Board:
    cells = np.asarray(cells, dtype=np.int8)
    return Board(cells.shape[0], cells.shape[1], cells.copy(), mines_left)


@dataclass
class Live:
    """Everything the probe needs for one synthetic site, all of it driven by pixels."""

    mf: Minefield
    perc: Perception
    board: Board
    frames: dict[bytes, np.ndarray]


def section(title: str) -> None:
    print(f"\n== {title} " + "=" * max(0, 58 - len(title)))


# --------------------------------------------------------------------------- #
# 1. settle
# --------------------------------------------------------------------------- #
class ScriptedScreen:
    """Returns rendered boards in order; a call past the end repeats the last one."""

    def __init__(self, boards: list[np.ndarray], *, unreadable_until: int = 0) -> None:
        self.boards = boards
        self.unreadable_until = unreadable_until
        self.reads = 0
        self.captures = 0

    def capture(self) -> np.ndarray:
        self.captures += 1
        return render_board(self.boards[min(self.captures - 1, len(self.boards) - 1)])

    def board_at(self, frame: np.ndarray) -> Board:
        self.reads += 1
        if self.reads <= self.unreadable_until:
            from ms.board import Unreadable
            raise Unreadable(f"read {self.reads} is mid-animation")
        return board_of(self.boards[min(self.reads - 1, len(self.boards) - 1)], 5)


def check_settle() -> bool:
    section("settle: two identical reads, the gap between them, and a timeout")
    ok = True

    a = np.full((5, 5), COVERED, np.int8)
    b = a.copy()
    b[2, 2] = 3

    s = ScriptedScreen([a, b, b, b, b], unreadable_until=2)
    with Clock() as clk:
        m = ex.wait_stable(s, gap=0.05, poll=0.01)
        slept = sum(clk.sleeps)
    settled_at = int(m.board.cells[2, 2])
    print(f"  two unreadable frames then a 3 -> settled on read {s.reads}, cell (2,2)={settled_at}, "
          f"waited {slept * 1000:.0f} ms")
    ok &= settled_at == 3 and slept >= 0.05

    class Ticker:
        n = 0

        def capture(self):
            self.n += 1
            return render_board(a)

        def board_at(self, frame):
            return board_of(a, self.n)      # counter ticks on every read

    with Clock():
        try:
            ex.wait_stable(Ticker(), timeout=0.5, gap=0.05, poll=0.01)
            print("  a board that changes every read -> settled (WRONG)")
            ok = False
        except ex.TimedOut as e:
            print(f"  a board that changes every read -> TimedOut: {e}")
    return ok


# --------------------------------------------------------------------------- #
# 2. after a move
# --------------------------------------------------------------------------- #
def check_after_move() -> bool:
    section("after a move: a click that did not land is named, not repeated")
    cells = np.full((5, 5), COVERED, np.int8)
    cells[2, 2] = 1
    before = board_of(cells, 9)

    class Stuck(ScriptedScreen):
        pass

    s = Stuck([cells])
    with Clock():
        try:
            ex.read_after_move(s, before, opened=[(2, 3)], gap=0.0)
            print("  nothing changed -> no error (WRONG)")
            return False
        except ex.Stuck as e:
            print(f"  nothing changed -> Stuck: {e}")

    moved = cells.copy()
    moved[2, 3] = 1
    with Clock():
        m = ex.read_after_move(ScriptedScreen([cells, moved]), before, opened=[(2, 3)], gap=0.0)
    # The screen reports the frames in order, so settling at `gap=0` lands on the second
    # read: `before` (first) -> `moved` (second) is exactly the change that was planned.
    ch = ex.confirm(before, m.board, expected=[(2, 3)])
    print(f"  expected cell changed -> {m.board.cells[2, 3]} at (2,3), confirm.ok={ch.ok}")
    return int(m.board.cells[2, 3]) == 1


# --------------------------------------------------------------------------- #
# 3. a live board through the real reader
# --------------------------------------------------------------------------- #
def live_board(seed: int = 3) -> Live:
    """An expert board in a state a chord probe can work on, read back through perception.

    The engine plays until a revealed number has as many flags round it as it shows - the
    exact precondition of a chord - and the board is then read from rendered pixels.
    """
    cfg = EXPERT
    mf = Minefield(cfg, np.random.default_rng(seed))
    rng = np.random.default_rng(seed + 1)
    idx = (cfg.rows // 2) * cfg.cols + cfg.cols // 2
    mf.click(idx)
    nb = mf.nb

    def chordable_numbers():
        for r in range(cfg.rows):
            for c in range(cfg.cols):
                i = r * cfg.cols + c
                v = int(mf.visible[i])
                if v < 0:
                    continue
                nbrs = [int(j) for j in nb[i] if j >= 0]
                flags = [j for j in nbrs if int(mf.visible[j]) == FLAG]
                hidden = [j for j in nbrs if int(mf.visible[j]) < 0]
                if len(flags) == v and hidden:
                    yield i, nbrs, hidden

    for _ in range(2000):
        # checks the *current* board; a restart reuses the same name, so re-bind `nb`
        found = next(chordable_numbers(), None)
        if found is not None:
            perc = Perception(Rect(THEME.margin, THEME.margin,
                                   cfg.cols * THEME.cell, cfg.rows * THEME.cell),
                              cfg.rows, cfg.cols)
            perc.calibrate(render_board(np.full((cfg.rows, cfg.cols), COVERED, np.int8)))
            seen = perc.read(render_board(mf.observation()), mf.mines_left_counter(), strict=False)
            if chord_candidates(seen):
                break
        # The reader cannot classify every glyph, so a board can be chordable to the engine
        # and not to what the screen shows: keep playing until both agree.
        # flag a proved mine, else click a proved-safe cell, else click a 50/50 and replay
        target = None
        for i in range(cfg.rows * cfg.cols):
            if int(mf.visible[i]) >= 0:
                nbrs = [int(j) for j in mf.nb[i] if j >= 0]
                hidden = [j for j in nbrs if int(mf.visible[j]) < 0]
                flags = [j for j in nbrs if int(mf.visible[j]) == FLAG]
                v = int(mf.visible[i])
                mines = [j for j in hidden if mf.mines[j]]
                if v - len(flags) == len(hidden) and mines and mines[0] not in flags:
                    target = ("flag", mines[0])
                    break
                if v == len(flags) and hidden:
                    safe = [j for j in hidden if not mf.mines[j]]
                    if safe:
                        target = ("click", safe[0])
                        break
        if target is None:
            covered = np.flatnonzero(mf.visible < 0)
            if not covered.size:
                break
            pick = int(rng.choice(covered))
            if mf.mines.ravel()[pick]:
                mf = Minefield(cfg, np.random.default_rng(int(rng.integers(1 << 30))))
                mf.click(idx)
                continue
            target = ("click", pick)
        if target[0] == "flag":
            mf.flag(target[1])
        else:
            mf.click(target[1])
        if mf.exploded or mf.won:
            mf = Minefield(cfg, np.random.default_rng(int(rng.integers(1 << 30))))
            mf.click(idx)

    p, m = THEME.cell, THEME.margin
    perc = Perception(Rect(m, m, cfg.cols * p, cfg.rows * p), cfg.rows, cfg.cols)
    perc.calibrate(render_board(np.full((cfg.rows, cfg.cols), COVERED, np.int8)))
    board = perc.read(render_board(mf.observation()), mf.mines_left_counter(), strict=False)
    return Live(mf=mf, perc=perc, board=board, frames={})


def check_live() -> bool:
    section("a live board, read by ms.perception, planned against the engine")
    live = live_board()
    b = live.board
    revealed = int(np.sum(b.cells >= 0))
    unknown = int(np.sum((b.cells < COVERED) & (b.cells != FLAG)))
    print(f"  {b.rows}x{b.cols}: {revealed} revealed, {b.mines_left} mines left, "
          f"{unknown} unreadable cell(s)")
    truth = live.mf.observation()
    agree = int(np.sum(b.cells == truth))
    print(f"  cells identical to the engine: {agree}/{truth.size}")

    cands = chord_candidates(b)
    opens = [c for _, cells in cands for c in cells]
    acts = plan_actions(b, opens)
    chords = sum(a.kind == "chord" for a in acts)
    print(f"  chordable numbers: {len(cands)}; planned actions for their cells: {len(acts)} "
          f"({chords} chords, {len(acts) - chords} clicks)")

    # one planned chord, applied to the engine exactly as the executor would send it
    applied = 0
    for act in acts:
        if act.kind != "chord":
            continue
        r, c = act.cell
        status = live.mf.chord(r * b.cols + c)
        after = live.mf.observation()
        opened = int(np.sum((b.cells == COVERED) & (after >= 0)))
        print(f"  engine chord on {act.cell}: {status}, opened {opened} cell(s), "
              f"planned {len(act.opens)}")
        applied = opened
        break
    return apply_ok(b, acts, applied)


def apply_ok(board: Board, acts, opened: int) -> bool:
    planned = next((len(a.opens) for a in acts if a.kind == "chord"), 0)
    return planned > 0 and opened >= planned


# --------------------------------------------------------------------------- #
# 4. the chord probe against site models
# --------------------------------------------------------------------------- #
FLAGGED = None


def flagged_board() -> Board:
    global FLAGGED
    if FLAGGED is None:
        cells = np.array([[COVERED, COVERED, FLAG],
                          [1, FLAG, COVERED],
                          [COVERED, 1, COVERED]], dtype=np.int8)
        FLAGGED = Board(3, 3, cells, 1)
    return FLAGGED


class SiteMouse:
    """A mouse that applies each completed gesture to a site model, by form."""

    def __init__(self, accepts: dict[str, bool]) -> None:
        self.accepts = accepts
        self.batch: list[tuple] = []
        self.held: set[str] = set()
        self.pos = (0, 0)
        self.cells = flagged_board().cells.copy()
        self.animating = False

    @contextmanager
    def grabbed(self):
        try:
            yield self
        finally:
            self.held.clear()

    def move(self, x, y):
        self.pos = (int(x), int(y))

    def press(self, button):
        self.batch.append(("down", button))
        self.held.add(button)

    def release(self, button):
        self.batch.append(("up", button))
        self.held.discard(button)
        if not self.held:
            self._apply()

    def cursor(self):
        return self.pos

    def _apply(self):
        batch, self.batch = self.batch, []
        order = [b for k, b in batch if k == "down"]
        r = (self.pos[1] - THEME.margin) // THEME.cell
        c = (self.pos[0] - THEME.margin) // THEME.cell
        nbrs = [(rr, cc) for rr in range(3) for cc in range(3) if (rr, cc) != (r, c)]
        if order == ["middle"] and self.accepts.get("middle"):
            opened = [x for x in nbrs if self.cells[x] == COVERED]
        elif order == ["left", "right"] and self.accepts.get("left+right"):
            opened = [x for x in nbrs if self.cells[x] == COVERED]
        elif order and order[0] == "left":
            opened = [(r, c)] if self.cells[r, c] == COVERED else []
        else:
            opened = []
        self.animating = True
        for rr, cc in opened:
            self.cells[rr, cc] = 1


class SiteScreen:
    def __init__(self, mouse: SiteMouse) -> None:
        self.mouse = mouse
        self.reads = 0

    def capture(self):
        base = render_board(self.mouse.cells)
        if self.mouse.animating:
            self.mouse.animating = False
            base = base.copy()
            base[:3, :3] ^= 0xFF       # the overlay that makes the next read unreadable
        return base

    def board_at(self, frame):
        from ms.board import Unreadable
        self.reads += 1
        if not np.array_equal(frame[..., 0], render_board(self.mouse.cells)[..., 0]):
            raise Unreadable("mid-animation")
        return board_of(self.mouse.cells, 1)

    def cell_point(self, frame, cell):
        return (THEME.margin + (cell[1] + 0.5) * THEME.cell,
                THEME.margin + (cell[0] + 0.5) * THEME.cell)


def check_probe() -> bool:
    section("chord probe: what each site actually accepts")
    rows = [
        ("middle + left+right", {"middle": True, "left+right": True}),
        ("middle only", {"middle": True}),
        ("left+right only", {"left+right": True}),
        ("no chording", {}),
    ]
    ok = True
    for name, accepts in rows:
        mouse = SiteMouse(accepts)
        screen = SiteScreen(mouse)
        with Clock():
            support = ex.ChordProbe(
                screen, ex.Pointer(mouse), between_trials=0.0, settle={"gap": 0.0}
            ).run(name)
        verdict = support.method if support.ok else "none"
        expected = "middle" if accepts.get("middle") else (
            "left+right" if accepts.get("left+right") else "none")
        mark = "PASS" if verdict == expected else "FAIL"
        print(f"  {mark}  {name:<20s} probe says {verdict:<10s} {support.details}")
        ok &= verdict == expected
    return ok


# --------------------------------------------------------------------------- #
# 5. the real backend
# --------------------------------------------------------------------------- #
def check_windows_mouse(hwnd: int | None) -> bool:
    section("Windows SendInput backend (read-only: nothing is clicked)")
    mouse = ex.WindowsMouse()
    cur = mouse.cursor()
    fg = mouse.foreground_window()
    print(f"  DPI set, cursor read-back {cur}, foreground window {fg:#x}")
    inside = ex._window_at(mouse.user32, cur[0], cur[1]) if cur else 0
    print(f"  window under the cursor: {inside:#x}"
          + (f"  (is --hwnd: {inside == hwnd})" if hwnd else ""))
    print(f"  events injected: {mouse.sent}  (a read-only check)")
    return cur is not None and inside != 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hwnd", type=int, default=None,
                    help="optional: window id to compare against the window under the cursor")
    args = ap.parse_args(argv)

    results = [
        ("settle", check_settle()),
        ("after move", check_after_move()),
        ("live board", check_live()),
        ("chord probe", check_probe()),
    ]
    if sys.platform == "win32":
        results.append(("windows mouse", check_windows_mouse(args.hwnd)))

    section("summary")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
