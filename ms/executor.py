"""Board -> real mouse, and proof that the screen agreed.

The executor is the layer where a mistake costs a game rather than a few points, so it
is deliberately dumb, and every step is checked:

    gesture(action, screen, frame)  Action + board frame -> one Gesture (pixels + buttons)
    act(gesture, pointer)           send it: move, down, up, with a release guard
    wait_stable(screen)             capture until the picture stops moving
    read_after_move(screen, before) the same, then "did anything the move named change?"
    confirm(before, after, expected) which cells changed, and were they all expected?
    ChordProbe(...).run(site)       does *this* site chord, and with which form?

Two chord forms, and they are not interchangeable
-------------------------------------------------
* middle click - one button, down+up, nothing to interleave. Preferred.
* left+right   - press *both* down, then release both. Releasing left before right is a
  plain left click, which opens a cell without the flag check; on a board carrying a
  human flag that is an instant loss.

So `ChordProbe` never assumes: it presses the forms on a real board and only reports
"accepted" when a *later capture* shows the chord effect - several cells opened by one
gesture - rather than a click's effect.

Timing, and why it lives here
-----------------------------
Settle detection is in this module because "did the click land" is an execution fact,
not a perception one: capture -> read -> capture, accept only when two reads are identical
*and* they were taken at least ``gap`` apart. The gap is what separates "the animation is
between frames" from "the picture has finished changing". It is measured on the clock, not
on the board, because a held frame reads the same as a finished one.

Platforms
---------
Windows is implemented (``ctypes`` + ``SendInput``, physical pixels, per-monitor DPI aware).
macOS/Linux backends are not written: ``WindowsMouse`` raises ``UnsupportedPlatform`` there
instead of pretending, and any other ``Mouse`` implementation can be passed in.
"""
from __future__ import annotations

import ctypes
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Optional, Protocol, Sequence

import numpy as np

from .board import COVERED, FLAG, UNREADABLE, Board, Unreadable
from .chording import Action, chord_candidates

__all__ = [
    "Gesture",
    "Moment",
    "Mouse",
    "Pointer",
    "BoardChange",
    "BoardScreen",
    "ChordSupport",
    "ChordProbe",
    "PREFERENCE",
    "CHORD_BUTTONS",
    "ExecError",
    "FocusLost",
    "TimedOut",
    "Stuck",
    "Unverified",
    "UnsupportedPlatform",
    "WindowsMouse",
    "act",
    "confirm",
    "gesture",
    "read_after_move",
    "wait_stable",
]

Point = tuple[int, int]


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #
class ExecError(RuntimeError):
    """Base class: the executor refused to act or could not verify the result."""


class FocusLost(ExecError):
    """The window that owns the board lost the foreground; a click would land elsewhere."""


class TimedOut(ExecError):
    """The picture never stopped moving within the budget."""


class Stuck(ExecError):
    """The move landed on the screen but changed nothing the board can see."""


class Unverified(ExecError):
    """The board after the move is not the board the move was supposed to produce."""


class UnsupportedPlatform(ExecError):
    """No mouse backend for this OS (or without Windows ``SendInput``)."""

    def __init__(self, platform: str | None = None) -> None:
        super().__init__(
            f"no native mouse backend for {platform or sys.platform}; "
            "WindowsMouse implements SendInput/GetPhysicalCursorPos, other platforms "
            "must supply a Mouse implementation"
        )


# --------------------------------------------------------------------------- #
# ports - everything that touches the OS or the screen
# --------------------------------------------------------------------------- #
class Mouse(Protocol):
    def move(self, x: int, y: int) -> None:
        """Absolute cursor position, physical pixels."""
        ...

    def press(self, button: str) -> None: ...
    def release(self, button: str) -> None: ...

    def cursor(self) -> Point | None: ...

    @contextmanager
    def grabbed(self):
        """Buttons not yet released when the block exits are released, always."""
        ...


@dataclass(frozen=True)
class Pointer:
    """The real mouse, focused window included.

    ``owner`` is an opaque token for the window that should be in front - the agent
    passes something like the foreground HWND it saw when the board was being read.
    ``None`` disables the check.
    """

    mouse: Mouse
    owner: object | None = None
    foreground: Callable[[], object | None] | None = None


class BoardScreen(Protocol):
    """What the executor needs from the screen.

    ``board_at`` must return a Board on a frame that is readable and raise on one that
    is not; ``wait_stable`` treats every returned Board as a fact, so an unreadable
    animation frame has to raise rather than come back as a guess.
    """

    def capture(self) -> np.ndarray: ...

    def board_at(self, frame: np.ndarray) -> Board: ...

    def cell_point(self, frame: np.ndarray, cell: tuple[int, int]) -> Point: ...


Screen = BoardScreen | Callable[[], np.ndarray]


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Gesture:
    """One physical gesture, already reduced to pixels."""

    kind: Literal["click", "chord"]
    point: Point
    cell: tuple[int, int]
    opens: tuple[tuple[int, int], ...]
    buttons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.buttons not in (("left",), ("middle",), ("left", "right")):
            raise ExecError(f"no gesture with buttons {self.buttons}")


def _cell_point(screen, frame: np.ndarray, cell: tuple[int, int]) -> Point:
    pt = screen.cell_point(frame, cell)
    x, y = (int(round(float(v))) for v in pt)
    if x < 0 or y < 0:
        raise ExecError(f"cell {cell} maps to negative screen coordinates ({x},{y})")
    return x, y


def gesture(action: Action, screen: BoardScreen, frame: np.ndarray, method: str = "middle") -> Gesture:
    """An Action plus a board frame -> pixels plus buttons. No I/O, fully testable."""
    if screen is None or not hasattr(screen, "cell_point"):
        raise TypeError("act() needs a BoardScreen to map cells to pixels")
    if action.kind == "open":
        return Gesture(
            "click", _cell_point(screen, frame, action.cell), action.cell, action.opens, ("left",)
        )
    if action.kind == "chord":
        if method not in CHORD_BUTTONS:
            raise ExecError(f"unknown chord method {method!r}; use one of {sorted(CHORD_BUTTONS)}")
        if len(action.opens) < 2:
            raise ExecError(
                f"chord on {action.cell} opens {len(action.opens)} cell(s); that is just a "
                "click with the flag check, use Action('open', ...) instead"
            )
        return Gesture(
            "chord", _cell_point(screen, frame, action.cell), action.cell, action.opens,
            CHORD_BUTTONS[method],
        )
    raise ExecError(f"{action.kind!r} is not a mouse action")


CHORD_BUTTONS: dict[str, tuple[str, ...]] = {
    "middle": ("middle",),
    "left+right": ("left", "right"),
}


# --------------------------------------------------------------------------- #
# actions
# --------------------------------------------------------------------------- #
def act(g: Gesture, where: Pointer) -> None:
    """Perform one gesture. Moves the cursor first so hover effects match the click."""
    if where.foreground is not None and where.owner is not None:
        now = where.foreground()
        if now != where.owner:
            raise FocusLost(
                f"window owner {where.owner!r} is not in front (foreground is {now!r}); "
                "refusing to click"
            )
    mouse = where.mouse
    # The first call positions the cursor only, so hover effects are identical for every
    # form; the buttons are then spelled out explicitly below. Which buttons a chord
    # uses is a per-site fact, not a preference - see `ChordProbe`.
    mouse.move(g.point[0], g.point[1])
    # `grabbed` releases anything still down if a call in between raises: a button stuck
    # down is a stuck user session. A backend without it has to refuse rather than run
    # two presses with no way back.
    if not hasattr(mouse, "grabbed"):
        raise ExecError("Mouse backend has no grabbed() guard; refusing to press buttons")
    with mouse.grabbed():
        if len(g.buttons) > 1 and hasattr(mouse, "pressed_together"):
            # Down-down in one SendInput call, up-up in another: no Python runs between
            # the presses, so not even an exception can turn this into a left click.
            mouse.pressed_together(g.buttons)
            mouse.released_together(g.buttons)
            return
        for b in g.buttons:
            mouse.press(b)
        for b in g.buttons:
            mouse.release(b)


# --------------------------------------------------------------------------- #
# settle + re-read
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Moment:
    """Two identical consecutive reads: the picture has stopped moving."""

    board: Board
    frames: tuple[np.ndarray, ...]
    capture_ms: float


def _capture(screen: Screen) -> np.ndarray:
    frame = screen.capture() if hasattr(screen, "capture") else screen()
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ExecError(f"screen returned {getattr(frame, 'dtype', type(frame))}, not uint8 pixels")
    return frame


def _read(screen: BoardScreen, frame: np.ndarray) -> Board:
    return screen.board_at(frame)


def _frame_token(board: Board) -> bytes:
    """Cells plus the mine counter as one comparable value: did the picture change?"""
    return board.cells.tobytes() + b"|" + str(board.mines_left).encode()


def _identical(a: Board, b: Board) -> bool:
    return a.rows == b.rows and a.cols == b.cols and _frame_token(a) == _frame_token(b)


def wait_stable(
    screen: BoardScreen,
    *,
    timeout: float = 1.5,
    gap: float = 0.05,
    poll: float = 0.01,
) -> Moment:
    """Capture until two consecutive reads agree, then return that board.

    The gap between the two captures is the whole point: an animation that is halfway
    done reads the same as the frame before it, but it does not read the same twice a
    few centiseconds later. Costs one extra read per attempt.
    """
    if gap < 0 or poll < 0 or timeout <= 0:
        raise ValueError("need timeout > 0 and non-negative gap/poll")
    t0 = time.perf_counter()

    def expired() -> bool:
        return time.perf_counter() - t0 >= timeout

    prev: tuple[np.ndarray, Board] | None = None
    prev_at: float | None = None
    while True:
        frame = _capture(screen)
        try:
            board = _read(screen, frame)
        except Unreadable:
            # Mid-animation pixels are unreadable by contract; that is the normal case the
            # loop exists for, so it resets the evidence instead of escaping.
            prev, prev_at = None, None
        else:
            now = time.perf_counter()
            # Float clocks: 1000.10 - 1000.05 is 0.0499999999999545, not 0.05. A settle
            # must not hinge on that, so the gap is compared with a rounding allowance far
            # below any real frame interval.
            waited = now - prev_at if prev_at is not None else 0.0
            settled = prev is not None and _identical(prev[1], board) and (
                waited + 1e-6 >= gap
            )
            if settled:

                return Moment(board, (prev[0], frame), 1000.0 * (now - t0))
            prev, prev_at = (frame, board), now
        if expired():
            raise TimedOut(
                f"no two identical reads within {timeout:.2f}s"
                + (" (the most recent frame was unreadable)" if prev is None else "")
            )
        # Wait out the rest of the gap, so the next frame is not the same instant as this
        # one; a shorter poll would only re-read an animation that has not moved yet.
        time.sleep(max(poll, gap) if prev is not None else poll)


def read_after_move(
    screen: BoardScreen,
    before: Board,
    *,
    opened: Iterable[tuple[int, int]] = (),
    timeout: float = 1.5,
    gap: float = 0.05,
    poll: float = 0.01,
) -> Moment:
    """The board *after* a move, once it has stopped changing.

    ``opened`` are the cells the move claimed it would open. If the settled board shows
    none of them opened, the move did not land (a swallowed click, a lost focus, or the
    board itself reset) - that is ``Stuck``, and it must not be papered over with another
    click. Real boards also draw hover highlights and "+mines" ticks, which is why this
    waits for stability rather than checking the expected board immediately.
    """
    targets = {(int(r), int(c)) for r, c in opened}
    moment = wait_stable(screen, timeout=timeout, gap=gap, poll=poll)
    after = moment.board
    if targets and after.rows == before.rows and after.cols == before.cols:
        if all(int(after.cells[r, c]) == int(before.cells[r, c]) for r, c in targets):
            raise Stuck(
                f"none of {len(targets)} expected cell(s) changed after the move "
                f"(e.g. {sorted(targets)[0]} is still {int(after.cells[sorted(targets)[0]])})"
            )
    return moment


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BoardChange:
    """What a move did to the picture, in board terms."""

    changed: tuple[tuple[int, int, int, int], ...]   # (r, c, before, after)
    expected_changed: tuple[tuple[int, int, int, int], ...]
    count_ok: bool | None
    appeared: tuple[tuple[int, tuple[int, int], int], ...] = ()   # (count, cell, value)

    @property
    def ok(self) -> bool:
        return not self.changed and not self.appeared and self.count_ok is not False


def confirm(before: Board, after: Board,
            expected: Optional[Iterable[tuple[int, int]]] = None) -> BoardChange:
    """Every changed cell must be one the move claimed to open.

    A cell that changed although the move never named it is the one failure mode worth
    catching before the agent plans the next move: it means the action opened more (or
    less) than the plan said, so the plan is no longer a description of the screen. The
    returned object is a fact, not a verdict - ``ok`` is a convenience.
    """
    if (before.rows, before.cols) != (after.rows, after.cols):
        raise Unverified(
            f"board geometry changed {before.rows}x{before.cols} -> "
            f"{after.rows}x{after.cols}: the game restarted or the rect moved"
        )
    allowed = None if expected is None else {(int(r), int(c)) for r, c in expected}
    changed: list[tuple[int, int, int, int]] = []
    unexpected: list[tuple[int, int, int, int]] = []
    appeared: list[tuple[int, tuple[int, int], int]] = []
    for r, c in np.argwhere(before.cells != after.cells):
        b, a = int(before.cells[r, c]), int(after.cells[r, c])
        changed.append((int(r), int(c), b, a))
        if allowed is not None and (int(r), int(c)) not in allowed:
            unexpected.append((int(r), int(c), b, a))
        if a == UNREADABLE or b == UNREADABLE:
            appeared.append((before.mines_left, (int(r), int(c)), a))
    count_ok = None if allowed is None else before.mines_left == after.mines_left
    # The mine counter only moves on a flag toggle; an open leaves it alone.
    if before.mines_left != after.mines_left and not any(a == FLAG or b == FLAG for _, _, b, a in changed):
        count_ok = False
    return BoardChange(
        changed=tuple(unexpected),
        expected_changed=tuple(x for x in changed if x not in unexpected),
        count_ok=count_ok,
        appeared=tuple(appeared),
    )


# --------------------------------------------------------------------------- #
# Windows backend - the only place that emits real input
# --------------------------------------------------------------------------- #
_MOUSEEVENTF = {"left": 0x0002, "right": 0x0008, "middle": 0x0020}
_BUTTONS = ("left", "right", "middle")
_INPUT_MOUSE = 0
_TPM_RETURNCMD = 0x0100


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long), ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong), ("wParamL", ctypes.c_ushort),
                ("wParamH", ctypes.c_ushort)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT), ("hi", _HARDWAREINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("u", _INPUTUNION)]


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class WindowsMouse:
    """``SendInput`` mouse in physical pixels, with a crash-proof button hold.

    Physical pixels matter: the session runs at 150% (GetDpiForSystem = 144), so logical
    coordinates from `GetSystemMetrics` are 1707x1067 where the capture is 2560x1600.
    `SetProcessDpiAwareness(PER_MONITOR_AWARE_V2)` before any user32 call is what makes
    `GetPhysicalCursorPos` - and therefore this backend's own feedback - agree with the
    screenshot.

    Crash safety: pressing both chord buttons one call after another is only safe if
    nothing in between can return. Real ``SendInput`` calls for down-down are queued
    back to back with no Python between them, and ``grabbed()`` releases whatever is
    still down on the way out - including on KeyboardInterrupt.
    """

    def __init__(self, *, dpi_aware: bool = True) -> None:
        if sys.platform != "win32":
            raise UnsupportedPlatform(sys.platform)
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        if not hasattr(self.user32, "SendInput"):
            raise UnsupportedPlatform("win32 without SendInput")
        if dpi_aware:
            try:
                # PROCESS_PER_MONITOR_DPI_AWARE (1) is enough to break logical scaling;
                # V2 is better where shcore exists.
                ctypes.WinDLL("shcore").SetProcessDpiAwareness(2)
            except (OSError, AttributeError):
                self.user32.SetProcessDPIAware()
        self._down: set[str] = set()
        self.sent = 0   # injected events, for tests and diagnostics

    # -- input -------------------------------------------------------------
    def _send(self, events: tuple[tuple[int, int], ...]) -> None:
        buf = (_INPUT * len(events))()
        for i, (flags, data) in enumerate(events):
            buf[i].type = _INPUT_MOUSE
            buf[i].u.mi = _MOUSEINPUT(0, 0, data, flags, 0, None)
        n = self.user32.SendInput(len(events), buf, ctypes.sizeof(_INPUT))
        if n != len(events):
            raise OSError(ctypes.get_last_error(), f"SendInput sent {n}/{len(events)} events")
        self.sent += len(events)

    def move(self, x: int, y: int) -> None:
        if self.cursor() == (int(x), int(y)):
            return
        self.user32.SetCursorPos(int(x), int(y))

    def press(self, button: str) -> None:
        self._send(((_MOUSEEVENTF[button], 0),))
        self._down.add(button)

    def release(self, button: str) -> None:
        self._send(((_MOUSEEVENTF[button] << 1, 0),))
        self._down.discard(button)

    def pressed_together(self, buttons: Sequence[str]) -> None:
        """Both downs in one SendInput call: nothing can interleave, not even a raise."""
        self._send(tuple((_MOUSEEVENTF[b], 0) for b in buttons))
        self._down.update(buttons)

    def released_together(self, buttons: Sequence[str]) -> None:
        self._send(tuple((_MOUSEEVENTF[b] << 1, 0) for b in buttons))
        self._down.difference_update(buttons)

    def release_all(self) -> None:
        """Release anything still held; safe to call from a ``finally``."""
        held = [b for b in _BUTTONS if b in self._down]
        if held:
            self.released_together(held)

    @contextmanager
    def grabbed(self):
        try:
            yield self
        finally:
            self.release_all()

    # -- read-back ---------------------------------------------------------
    def cursor(self) -> Point | None:
        pt = _POINT()
        if not self.user32.GetPhysicalCursorPos(ctypes.byref(pt)):
            return None
        return int(pt.x), int(pt.y)

    def foreground_window(self) -> int:
        return int(self.user32.GetForegroundWindow())


def _window_at(user32, x: int, y: int) -> int:
    """Top-level window under a point, and its root owner.

    Minesweeper ports are embedded in a top-level window; the point landed on a child
    window, so the foreground check compares root owners rather than window handles.
    """
    hwnd = user32.WindowFromPoint(_POINT(x, y))
    if not hwnd:
        return 0
    hwnd = user32.GetAncestor(hwnd, 2)   # GA_ROOT
    owner = user32.GetAncestor(hwnd, 3)  # GA_ROOTOWNER
    return int(owner or hwnd)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# chording support probe
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChordSupport:
    """What one probe learned about a site's chording.

    ``accepted`` decides which form the agent may use; the rest is the evidence.
    ``first_drag`` records the left-then-right ordering when nothing else worked: if it
    produced the *click* effect instead of the chord effect, that is the ordering trap
    made visible, not a reason to retry it.
    """

    site: str
    tried: tuple[str, ...]
    accepted: tuple[str, ...]
    first_drag: str | None
    details: dict[str, str]

    @property
    def ok(self) -> bool:
        return bool(self.accepted)

    @property
    def method(self) -> str:
        """The form to use: the preferred accepted one. Raises if there is none."""
        for m in PREFERENCE:
            if m in self.accepted:
                return m
        raise ExecError(
            f"{self.site}: no chord form accepted "
            f"({'; '.join(self.details.values()) or 'no detail'}); play with plain clicks"
        )

    def summary(self) -> str:
        if not self.ok:
            return f"{self.site}: no chord accepted - " + "; ".join(self.details.values())
        head = f"{self.site}: chord via {self.method}"
        if self.first_drag == "click":
            return head + "; left-then-right degrades to a click, only a together-press is safe"
        return head


PREFERENCE = ("middle", "left+right")


@dataclass(frozen=True)
class _Trial:
    target: tuple[int, int]
    opens: tuple[tuple[int, int], ...]
    point: Point


def _trials(board: Board, point_of: Callable[[tuple[int, int]], Point],
            limit: int = 12) -> list[_Trial]:
    """Numbers whose flags already match their value: pressing them is legal *now*.

    A chord on one opens two or more cells; a plain click on the same number opens
    nothing at all, because it is already revealed. That asymmetry is what makes the
    probe's verdict unambiguous. Candidates are ranked by how many cells they would open,
    and each is preferred over candidates sharing a cell with it: a multi-cell chord gives
    the verdict more to stand on, and one trial should not decide the next trial's effect.
    A candidate next to an unreadable cell is skipped: the site could refuse a chord for
    the flag count rather than for the button.
    """
    ranked = sorted(
        (c for c in chord_candidates(board)
         if not any(int(board.cells[r, c2]) == UNREADABLE for r, c2 in c[1])),
        key=lambda c: (-len(c[1]), c[0]),
    )
    taken: set[tuple[int, int]] = set()
    out: list[_Trial] = []
    for cell, opens in ranked:
        if opens[0] in taken:
            continue
        taken.update(opens)
        out.append(_Trial(cell, opens, point_of(cell)))
        if len(out) >= limit:
            break
    return out


def _effect(before: Board, after: Board, trial: _Trial) -> Literal["chord", "click", "nothing", "reset"]:
    """How the picture moved, in the vocabulary the verdict is made of.

    Playing a non-chord into a real game is not inert in general (a click that lands on a
    covered cell reveals it), so this measures the picture rather than asserting what a
    gesture "should" have done:

    * chord   - two or more cells changed. A number never turns back into a covered cell
                and never changes value, so only a chord - a single gesture opening several
                neighbours - can do this;
    * click   - exactly one cell changed;
    * nothing - no change at all;
    * reset   - the grid shape changed (the game restarted or the rect moved).

    Deliberately *not* in this list: "a cell changed that the plan did not name". The
    left-then-right form can straddle two chords and open a cell that the current trial
    does not mention; that is still a legible picture (one vs several cells), and the
    probe's job is to read it, not to outlaw it. The caller audits plans separately.
    """
    if (before.rows, before.cols) != (after.rows, after.cols):
        return "reset"
    diff = {
        (int(r), int(c)) for r, c in np.argwhere(before.cells != after.cells)
    }
    planned = set(trial.opens)
    if not diff:
        return "nothing"
    return "chord" if len(diff) >= 2 else "click"


class ChordProbe:
    """Decide, on a real board, whether this site chords and with which button.

    Order: middle click, then left+right with both presses queued before either release,
    then - only if nothing was accepted - left-then-right, documented once. Where a site
    chords at all, the last form is expected to land as a click; that row is the answer
    "use the middle button here", not a retry budget.

    The probe never infers from "the gesture was sent": only the picture decides. After a
    cell opens it no longer numbers anything, so the probe re-plans from the new board and
    never presses a target the site has already revealed. A change the gesture cannot
    explain (see ``_effect``) aborts the probe instead of being recorded as a success.
    """

    def __init__(
        self,
        screen: BoardScreen,
        where: Pointer,
        *,
        guard: Callable[[], None] | None = None,
        settle: dict | None = None,
        between_trials: float = 0.15,
        per_target: int = 2,
    ) -> None:
        self.screen = screen
        self.where = where
        self.guard = guard
        self.settle = {"timeout": 1.5, "gap": 0.05, "poll": 0.01, **(settle or {})}
        self.between_trials = between_trials
        self.per_target = per_target

    # -- one gesture per form ---------------------------------------------
    def _gesture(self, method: str, trial: _Trial) -> None:
        if self.guard is not None:
            self.guard()
        buttons = ("left",) if method == "left" else CHORD_BUTTONS[method]
        act(Gesture("click", trial.point, trial.target, trial.opens, buttons), self.where)

    # -- the probe ---------------------------------------------------------
    def run(self, site: str, *, first_drag: bool = True) -> ChordSupport:
        before = self._settle().board
        sealed = self._sealed_cells(before)
        if not self._plan(before):
            raise ExecError(
                f"{site}: nothing chordable on this board - no revealed number has flags "
                "equal to its value; probe after a few moves, not on a fresh board"
            )

        details: dict[str, str] = {}
        accepted: list[str] = []
        used: dict[tuple[int, int], int] = {}

        for method in PREFERENCE:
            attempted = 0
            for trial in self._plan(before):
                if used.get(trial.target, 0) >= self.per_target:
                    continue
                used[trial.target] = used.get(trial.target, 0) + 1
                attempted += 1
                self._gesture(method, trial)
                board = self._settle().board
                effect = _effect(before, board, trial)
                details[method] = f"{effect} on {trial.target}"
                if effect == "reset":
                    raise ExecError(
                        f"{site}: board geometry changed during the {method} probe "
                        f"({details[method]}); the game restarted or the rect moved - re-probe"
                    )
                self._audit(site, method, before, board, trial, sealed)
                before = board
                if effect == "chord":
                    accepted.append(method)
                    break
                if self.between_trials:
                    time.sleep(self.between_trials)
            if not attempted:
                details[method] = "not tried (no chordable number left)"
            if accepted:
                break

        drag = None
        if first_drag and not accepted:
            for trial in self._plan(before):
                if used.get(trial.target, 0) >= self.per_target:
                    continue
                used[trial.target] = used.get(trial.target, 0) + 1
                self._gesture("left", trial)
                board = self._settle().board
                drag = _effect(before, board, trial)
                details["left-then-right"] = f"{drag} on {trial.target}"
                if drag == "chord":
                    accepted.append("left-then-right")
                break

        return ChordSupport(site, tuple(details), tuple(accepted), drag, dict(details))

    @staticmethod
    def _sealed_cells(board: Board) -> set[tuple[int, int]]:
        """Cells that already contradict the numbers around them.

        A wrong board never becomes right by clicking: if a number's flags exceed it, no
        gesture can repair that. Such cells are the one place where "it changed" is not
        evidence, and they are excluded from the audit - (1,1) below is a flag with zero
        neighbours, a number that is false from the first frame.
        """
        rows, cols = board.rows, board.cols
        sealed: set[tuple[int, int]] = set()
        for r in range(rows):
            for c in range(cols):
                nbrs = [(rr, cc) for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)
                        if (rr, cc) != (r, c) and 0 <= rr < rows and 0 <= cc < cols]
                v = int(board.cells[r, c])
                flags = sum(1 for rr, cc in nbrs if int(board.cells[rr, cc]) == FLAG)
                if v >= 0 and flags > v:
                    # the number is false; its neighbourhood and itself are unpoliced
                    sealed.update(nbrs)
                    sealed.add((r, c))

        return sealed

    def _audit(self, site: str, method: str, before: Board, after: Board, trial: _Trial,
               sealed: set[tuple[int, int]]) -> None:
        """Abort if the gesture opened a cell that no gesture could reach from here.

        "Reachable" is deliberately generous - the target's whole 3x3 neighbourhood, plus
        any region that a zero flood could have opened from it. What is left over is the
        failure this probe exists to catch: the board changed somewhere the click cannot
        possibly have acted (a restart behind the same geometry, a rect that slid, a
        misread frame). Cells that already contradict their own flags are excluded, since
        no gesture can repair those and counting them would abort a legitimate probe.
        """
        r, c = trial.target
        reached = {(rr, cc) for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)}

        def flood(start: tuple[int, int], cells: np.ndarray) -> set[tuple[int, int]]:
            # cells a click/chord could open through a zero neighbour: BFS over cells that
            # were covered before the gesture and are open after it, adjacent to a zero.
            rows, cols = cells.shape
            frontier = [start]
            seen: set[tuple[int, int]] = {start}
            while frontier:
                rr, cc = frontier.pop()
                for r2 in range(rr - 1, rr + 2):
                    for c2 in range(cc - 1, cc + 2):
                        if not (0 <= r2 < rows and 0 <= c2 < cols) or (r2, c2) in seen:
                            continue
                        seen.add((r2, c2))
                        if int(before.cells[r2, c2]) != COVERED:
                            continue
                        if int(after.cells[r2, c2]) == 0:
                            frontier.append((r2, c2))
            return seen

        reached |= flood((r, c), after.cells)
        # one ring of slack: a chord can start a flood chain from any cell it opened.
        edge = {(rr, cc) for cr, cc0 in reached
                for rr in range(cr - 1, cr + 2) for cc in range(cc0 - 1, cc0 + 2)}
        reached |= edge
        changed = {(int(rr), int(cc)) for rr, cc in np.argwhere(before.cells != after.cells)}
        # A revealed cell never goes back to covered and its number never gets smaller;
        # either one means this is a different board.
        returned = {
            cell for cell in changed
            if int(before.cells[cell]) >= 0
            and (int(after.cells[cell]) < 0 or int(after.cells[cell]) < int(before.cells[cell]))
            and cell not in sealed
        }
        if returned:
            raise ExecError(
                f"{site}: the {method} probe on {trial.target} turned "
                f"{sorted(returned)[:4]} from open to a smaller number; nothing on a "
                "minesweeper board does that - re-probe"
            )
        strange = {cell for cell in changed if cell not in reached and cell not in sealed}
        if strange:
            raise ExecError(
                f"{site}: the {method} probe on {trial.target} changed {sorted(strange)[:4]} "
                "outside any chord or click that cell can cause; the board is not the one "
                "being probed (restart behind the same geometry, moved rect, or a misread) "
                "- re-probe"
            )

    def _settle(self) -> Moment:
        return wait_stable(self.screen, **self.settle)

    def _plan(self, board: Board) -> list[_Trial]:
        frame = _capture(self.screen)
        return _trials(board, lambda cell: _cell_point(self.screen, frame, cell))
