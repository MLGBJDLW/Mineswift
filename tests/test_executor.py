"""Executor: the mouse, the settle loop, the re-read, and the per-site chord probe.

Everything here runs against fakes on purpose. The one property that must hold before a
real click is ever emitted:

    the executor does not believe a gesture happened - it believes the picture.

So the tests drive a fake mouse that *applies* the gesture to a real `Minefield`, render
the result, and let the real `ms.perception` reader produce the Board. Where a test needs
the click to be wrong (stuck, rejected chord, degraded left click) the fake model is told
to drop it, and the executor must notice.

Real input is never injected here; `experiments/executor_probe.py --hwnd` is the only path
that touches ``SendInput``.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
import pytest

from ms import executor as ex
from ms.board import COVERED, FLAG, Board, Unreadable
from ms.chording import Action, chord_candidates, plan_actions
from ms.engine import Config, Minefield
from ms.perception import Perception, Rect

from tests._synthetic import Theme, render

THEME = Theme(
    cell=30,
    covered=(176, 178, 184), covered_light=(214, 216, 222), covered_dark=(132, 134, 140),
    bg=(198, 200, 206), ink=(30, 34, 44), grid=(96, 98, 104),
    flag=(188, 40, 44), margin=12,
)


def _render(cells: np.ndarray) -> np.ndarray:
    img, _ = render(cells, THEME)
    return img


def find_subimage(hay: np.ndarray, needle: np.ndarray) -> tuple[int, int] | None:
    """Top-left offset where ``needle`` appears in ``hay``, or None. Exact, grey-space."""
    hh, hw = hay.shape[:2]
    nh, nw = needle.shape[:2]
    if (nh, nw) > (hh, hw):
        return None
    g = hay[..., 0].astype(np.int16)
    q = needle[..., 0].astype(np.int16)
    # cheap filter: candidate x positions share the needle's first row
    xs = [x for x in range(hw - nw + 1) if np.array_equal(g[0, x:x + nw], q[0])]
    for y in range(hh - nh + 1):
        for x in xs:
            if np.array_equal(g[y:y + nh, x:x + nw], q):
                return x, y
    return None


def read_region(frame: np.ndarray, candidates, origin: tuple[int, int]) -> Board:
    """What a real reader does with a rect: find *which* board the screenshot shows.

    Perception is given the user's rect - a fixed area of the screen - and picks the
    board whose pixels it finds inside it. Anything unreadable is `Unreadable`, exactly
    like the real reader on a frame that is mid-animation or the wrong region.
    """
    for cells in candidates:
        if find_subimage(frame, _render(cells)) is not None:
            return _board(cells, 3)
    raise Unreadable("no readable board in the frame")


def _board(cells: np.ndarray, mines_left: int = 9, frame_id: int = 0) -> Board:
    cells = np.asarray(cells, dtype=np.int8)
    return Board(cells.shape[0], cells.shape[1], cells.copy(), mines_left, frame_id)


def _frame_token(board: Board) -> bytes:
    """The cells plus the counter as one comparable value: two reads, same picture?"""
    return board.cells.tobytes() + b"|" + str(board.mines_left).encode()


class FakeScreen:
    """A live board behind the BoardScreen port, plus control over *when* it is readable.

    It is faithful to the real reader in the one way the settle loop depends on: the same
    pixels give the same Board, and pixels the other screen left behind are unreadable
    (the real `Perception.read` raises on a frame it cannot classify, and it has no memory
    of earlier frames). The overlay marks a frame as mid-animation.
    """

    def __init__(self, cells: np.ndarray, *, mines_left: int = 9) -> None:
        self.cells = np.asarray(cells, dtype=np.int8).copy()
        self.mines_left = mines_left
        self.reads = 0
        self.captures = 0
        self.frame_id = 0
        self.unreadable_reads: set[int] = set()   # 1-based read numbers that raise
        self.overlay = False                      # one-shot: the next read is unreadable

    def capture(self) -> np.ndarray:
        self.captures += 1
        img = _render(self.cells)
        if self.overlay:
            img = img.copy()
            img[:4, :4] ^= 0xFF        # corner marker: not inside any cell
            self.overlay = False
        return img

    def board_at(self, frame: np.ndarray) -> Board:
        self.reads += 1
        if self.reads in self.unreadable_reads:
            raise Unreadable(f"read {self.reads} is mid-animation")
        if not np.array_equal(frame[..., 0], _render(self.cells)[..., 0]):
            raise Unreadable(f"read {self.reads}: frame is not the current board")
        return _board(self.cells, self.mines_left, self.frame_id)

    def cell_point(self, frame, cell):
        r, c = cell
        return THEME.margin + (c + 0.5) * THEME.cell, THEME.margin + (r + 0.5) * THEME.cell

    # test helpers --------------------------------------------------------
    def set(self, r: int, c: int, v: int) -> None:
        self.cells[r, c] = v
        self.frame_id += 1


class SimMouse:
    """Applies each completed gesture to a model, the way a site would.

    Events are collected per gesture: one completed gesture is the run of button events
    between a cursor move and the moment every button is up again. A model therefore
    sees exactly one chord or click per ``act``, never the leftovers of the last one.
    """

    def __init__(self, apply=None) -> None:
        self.apply = apply
        self.held: set[str] = set()
        self.pos = (0, 0)
        self.events: list[tuple] = []
        self.grabbed_entered = 0
        self._batch: list[tuple] = []
        self.batch_open = False

    @contextmanager
    def grabbed(self):
        self.grabbed_entered += 1
        try:
            yield self
        finally:
            self.held.clear()

    def move(self, x, y):
        self.events.append(("move", int(x), int(y)))
        self.pos = (int(x), int(y))

    def press(self, button):
        self.events.append(("down", button, tuple(sorted(self.held))))
        self.held.add(button)
        self.batch_open = True

    def release(self, button):
        self.events.append(("up", button, tuple(sorted(self.held))))
        self.held.discard(button)
        if not self.held and self.batch_open:
            self.batch_open = False
            self._complete()

    def _complete(self):
        batch, self.events, self._batch = self._batch + self.events, [], []
        if self.apply is not None:
            self._batch = batch
            self.apply(self)
            self._batch = []
        self.events = []

    def cursor(self):
        return self.pos


# --------------------------------------------------------------------------- #
# fake screen: rendered pixels, real perception, scripted unreadability
# --------------------------------------------------------------------------- #
class FakeScreen:
    """A live board behind the BoardScreen port, plus control over *when* it is readable.

    Reading follows the contract the real reader has today: pixels alone. ``overlay`` is
    the parts of the picture that are *not* cell content (a hover highlight, an animation
    layer, a tooltip); while it is set the read is `Unreadable`. A site model or a test
    turns it on for exactly one read to simulate a frame that had not settled yet.
    """

    def __init__(self, cells: np.ndarray, *, mines_left: int = 9) -> None:
        self.cells = np.asarray(cells, dtype=np.int8).copy()
        self.mines_left = mines_left
        self.reads = 0
        self.captures = 0
        self.frame_id = 0
        self.unreadable_reads: set[int] = set()   # 1-based read numbers that raise
        self.overlay = False                      # one-shot: the next read is unreadable

    # BoardScreen ---------------------------------------------------------
    def capture(self) -> np.ndarray:
        self.captures += 1
        img = _render(self.cells)
        if self.overlay:
            img = img.copy()
            img[:4, :4] ^= 0xFF        # corner marker: not part of any cell's content
            self.overlay = False
        return img

    def board_at(self, frame: np.ndarray) -> Board:
        self.reads += 1
        if self.reads in self.unreadable_reads:
            raise Unreadable(f"read {self.reads} is mid-animation")
        if not np.array_equal(frame[..., 0], _render(self.cells)[..., 0]):
            raise Unreadable(f"read {self.reads}: frame is not the current board")
        return _board(self.cells, self.mines_left, self.frame_id)

    def cell_point(self, frame, cell):
        r, c = cell
        return THEME.margin + (c + 0.5) * THEME.cell, THEME.margin + (r + 0.5) * THEME.cell

    # test helpers --------------------------------------------------------
    def set(self, r: int, c: int, v: int) -> None:
        self.cells[r, c] = v
        self.frame_id += 1


# --------------------------------------------------------------------------- #
# gesture: Action + frame -> pixels + buttons
# --------------------------------------------------------------------------- #
def test_open_becomes_a_left_click_at_the_cell_centre():
    s = FakeScreen(np.full((3, 3), COVERED, np.int8))
    g = ex.gesture(Action("open", (1, 2), ((1, 2),)), s, None)
    assert g.kind == "click" and g.buttons == ("left",)
    assert g.point == (12 + int((2 + 0.5) * 30), 12 + int((1 + 0.5) * 30))


@pytest.mark.parametrize("method, buttons", [("middle", ("middle",)), ("left+right", ("left", "right"))])
def test_chord_uses_the_sites_buttons(method, buttons):
    s = FakeScreen(np.full((3, 3), COVERED, np.int8))
    g = ex.gesture(Action("chord", (0, 0), ((0, 1), (1, 0))), s, None, method)
    assert g.kind == "chord" and g.buttons == buttons


def test_a_chord_that_opens_one_cell_is_refused():
    s = FakeScreen(np.full((3, 3), COVERED, np.int8))
    with pytest.raises(ex.ExecError, match="just a click"):
        ex.gesture(Action("chord", (0, 0), ((0, 1),)), s, None, "middle")


def test_unknown_method_and_unknown_action_are_refused():
    s = FakeScreen(np.full((3, 3), COVERED, np.int8))
    with pytest.raises(ex.ExecError, match="unknown chord method"):
        ex.gesture(Action("chord", (0, 0), ((0, 1), (1, 0))), s, None, "ctrl+click")
    with pytest.raises(ex.ExecError, match="not a mouse action"):
        ex.gesture(Action("flag", (0, 0), ((0, 0),)), s, None)


def test_gesture_needs_a_screen_that_can_map_cells():
    with pytest.raises(TypeError):
        ex.gesture(Action("open", (0, 0), ((0, 0),)), None, None)


# --------------------------------------------------------------------------- #
# act: order of events, focus check, no stuck buttons
# --------------------------------------------------------------------------- #
def test_a_chord_presses_both_before_releasing_either():
    class Wiring(SimMouse):
        """SimMouse applies on cursor-up, but this test reads the events themselves."""

        def _complete(self):
            pass

    mouse = Wiring()
    s = FakeScreen(np.full((3, 3), COVERED, np.int8))
    g = ex.gesture(Action("chord", (0, 0), ((0, 1), (1, 0))), s, None, "left+right")
    ex.act(g, ex.Pointer(mouse))
    buttons = [(kind, btn) for kind, btn, *_ in mouse.events if kind != "move"]
    assert buttons == [("down", "left"), ("down", "right"), ("up", "left"), ("up", "right")]
    assert mouse.held == set(), "nothing may still be pressed when act() returns"


def test_focus_loss_refuses_before_any_button_moves():
    mouse = SimMouse()
    s = FakeScreen(np.full((3, 3), COVERED, np.int8))
    g = ex.gesture(Action("open", (0, 0), ((0, 0),)), s, None)
    ptr = ex.Pointer(mouse, owner=7, foreground=lambda: 9)
    with pytest.raises(ex.FocusLost):
        ex.act(g, ptr)
    assert mouse.events == []


def test_an_owner_that_is_still_in_front_is_clicked():
    class Wiring(SimMouse):
        def _complete(self):
            pass

    mouse = Wiring()
    s = FakeScreen(np.full((3, 3), COVERED, np.int8))
    g = ex.gesture(Action("open", (0, 0), ((0, 0),)), s, None)
    ex.act(g, ex.Pointer(mouse, owner=7, foreground=lambda: 7))
    buttons = [(k, b) for k, b, *_ in mouse.events if k != "move"]
    assert buttons == [("down", "left"), ("up", "left")]
    assert mouse.held == set()


def test_a_failing_press_never_leaves_a_button_held():
    """A Mouse without ``grabbed`` is told to stop: no button, no half-pressed chord."""
    class Broken(SimMouse):
        def release(self, button):
            raise OSError("input queue full")

    mouse = Broken()
    s = FakeScreen(np.full((3, 3), COVERED, np.int8))
    g = ex.gesture(Action("chord", (0, 0), ((0, 1), (1, 0))), s, None, "left+right")
    with pytest.raises(OSError):
        ex.act(g, ex.Pointer(mouse))
    assert mouse.held == set(), "left down / right down order is not reached until grabs exist"


def test_a_backend_without_a_guard_is_refused_before_any_press():
    """No ``grabbed()`` means no way to undo a half-finished chord: refuse up front."""
    class Unguarded:
        def __init__(self):
            self.events = []

        def move(self, x, y):
            self.events.append(("move", x, y))

        def press(self, button):
            self.events.append(("down", button))

        def release(self, button):
            self.events.append(("up", button))

        def cursor(self):
            return 0, 0

    mouse = Unguarded()
    s = FakeScreen(np.full((3, 3), COVERED, np.int8))
    g = ex.gesture(Action("chord", (0, 0), ((0, 1), (1, 0))), s, None, "left+right")
    with pytest.raises(ex.ExecError, match="grabbed"):
        ex.act(g, ex.Pointer(mouse))
    assert [e for e in mouse.events if e[0] != "move"] == []


def test_the_legacy_release_path_runs_inside_grabbed():
    """When the backend has no batch press, ``grabbed()`` is what guarantees a clean state."""
    mouse = SimMouse()
    s = FakeScreen(np.full((3, 3), COVERED, np.int8))
    g = ex.gesture(Action("open", (0, 0), ((0, 0),)), s, None)
    ex.act(g, ex.Pointer(mouse))
    assert mouse.grabbed_entered == 1 and mouse.held == set()


def test_the_real_backend_is_used_when_it_can_press_two_buttons_together():
    class Together(SimMouse):
        def __init__(self):
            super().__init__()
            self.batches: list[tuple] = []

        def pressed_together(self, buttons):
            self.batches.append(("down", tuple(buttons)))

        def released_together(self, buttons):
            self.batches.append(("up", tuple(buttons)))
            self.held.clear()

    mouse = Together()
    s = FakeScreen(np.full((3, 3), COVERED, np.int8))
    g = ex.gesture(Action("chord", (0, 0), ((0, 1), (1, 0))), s, None, "left+right")
    ex.act(g, ex.Pointer(mouse))
    assert mouse.batches == [("down", ("left", "right")), ("up", ("left", "right"))]


class LoopingScreen(FakeScreen):
    """Changes on purpose: one screen for every test that must not depend on settling.

    Handing `wait_stable` a board that never changes would settle on its own - and in
    these tests sleep is instant, so a "never settles" timeout that *settles* would run
    forever rather than fail. A board that moves on every read makes the loop do exactly
    one thing: keep looking until the budget expires.
    """

    def board_at(self, frame: np.ndarray) -> Board:
        out = super().board_at(frame)
        self.set(int(self.reads % out.rows), 0, int(self.reads % 2))
        return out


# --------------------------------------------------------------------------- #
# wait_stable: the settle loop
# --------------------------------------------------------------------------- #
def test_settles_only_after_an_unreadable_read(fast_clock):
    s = FakeScreen(np.full((4, 4), COVERED, np.int8))
    s.unreadable_reads = {1, 2}
    m = ex.wait_stable(s, gap=0.05, poll=0.01)
    assert s.reads == 4 and m.board.rows == 4


def test_settle_waits_the_gap_between_the_two_identical_reads(fast_clock):
    s = FakeScreen(np.full((4, 4), COVERED, np.int8))
    ex.wait_stable(s, gap=0.05, poll=0.01)
    assert s.reads == 2
    assert fast_clock.sleeps == [0.05]


class LoopingScreen(FakeScreen):
    """Every read reports a different board: the loop must keep looking and then time out.

    The real analogue is an animation that never ends, or a game whose clock keeps painting.
    Handing `wait_stable` a board that never changes would settle at once, which is why this
    one inherits nothing that settles: its counter moves on every read.
    """

    def board_at(self, frame: np.ndarray) -> Board:
        super().board_at(frame)          # counts the read and rejects stale pixels
        self.set(self.reads % self.cells.shape[0], 0, self.reads % 2)
        return _board(self.cells, self.mines_left + self.reads, self.frame_id)


def test_a_board_that_keeps_changing_times_out(fast_clock):
    with pytest.raises(ex.TimedOut):
        ex.wait_stable(LoopingScreen(np.full((4, 4), COVERED, np.int8)),
                       timeout=0.2, gap=0.05, poll=0.01)


def test_an_increasing_mine_counter_is_a_change_not_a_pause(fast_clock):
    s = FakeScreen(np.full((4, 4), COVERED, np.int8))

    def bump(frame):
        s.mines_left += 1
        return _board(s.cells, s.mines_left)

    s.board_at = bump
    with pytest.raises(ex.TimedOut):
        ex.wait_stable(s, timeout=0.1, gap=0.05, poll=0.01)


def test_capture_must_be_uint8_pixels(fast_clock):
    class Bad(FakeScreen):
        def capture(self):
            return np.zeros((4, 4, 3), dtype=np.float32)

    with pytest.raises(ex.ExecError, match="not uint8"):
        ex.wait_stable(Bad(np.full((4, 4), COVERED, np.int8)), gap=0.0)


def test_a_bare_capture_callable_is_a_screen(fast_clock):
    cells = np.full((4, 4), COVERED, np.int8)
    seen = ex._capture(lambda: _render(cells))
    assert seen.dtype == np.uint8 and seen.shape[2] == 3


# --------------------------------------------------------------------------- #
# read_after_move + confirm
# --------------------------------------------------------------------------- #
def test_an_open_that_did_not_land_is_stuck_not_retried(fast_clock):
    s = FakeScreen(np.full((5, 5), COVERED, np.int8))
    s.set(2, 2, 0)
    before = _board(s.cells)
    with pytest.raises(ex.Stuck):
        ex.read_after_move(s, before, opened=[(2, 3)], timeout=0.05, gap=0.0, poll=0.0)


def test_an_open_that_landed_is_returned(fast_clock):
    s = FakeScreen(np.full((5, 5), COVERED, np.int8))
    s.set(2, 2, 0)
    before = _board(s.cells)
    s.set(2, 3, 1)
    m = ex.read_after_move(s, before, opened=[(2, 3)], gap=0.0)
    assert int(m.board.cells[2, 3]) == 1


def test_read_after_move_without_targets_is_just_a_settle(fast_clock):
    s = FakeScreen(np.full((5, 5), COVERED, np.int8))
    m = ex.read_after_move(s, _board(s.cells), gap=0.0)
    assert m.board.rows == 5


def test_confirm_separates_expected_from_unexpected_changes():
    before = np.full((3, 3), COVERED, np.int8)
    before[1, 1] = 1
    after = before.copy()
    after[0, 1] = 2      # planned
    after[2, 2] = 0      # appeared from nowhere
    ch = ex.confirm(_board(before), _board(after), expected=[(0, 1)])
    assert ch.changed == ((2, 2, COVERED, 0),)
    assert ch.expected_changed == ((0, 1, COVERED, 2),)
    assert ch.ok is False


def test_confirm_passes_when_every_change_was_expected():
    before = np.full((3, 3), COVERED, np.int8)
    after = before.copy()
    after[0, 0], after[1, 1] = 0, FLAG
    ch = ex.confirm(_board(before, 3), _board(after, 3), expected=[(0, 0), (1, 1)])
    assert ch.ok and not ch.changed


def test_confirm_has_no_expectation_without_a_plan():
    before = np.full((3, 3), COVERED, np.int8)
    after = before.copy()
    after[1, 1] = 5
    ch = ex.confirm(_board(before), _board(after))
    assert ch.ok and ch.expected_changed == ((1, 1, COVERED, 5),)


def test_confirm_rejects_a_different_board_geometry():
    with pytest.raises(ex.Unverified):
        ex.confirm(_board(np.full((3, 3), COVERED, np.int8)),
                   _board(np.full((9, 9), COVERED, np.int8)))


def test_confirm_flags_an_unexplained_mine_counter_move():
    before = np.full((3, 3), COVERED, np.int8)
    after = before.copy()
    after[0, 0] = 0
    ch = ex.confirm(_board(before, 10), _board(after, 9), expected=[(0, 0)])
    assert ch.count_ok is False and not ch.ok


# --------------------------------------------------------------------------- #
# the probe, against a site model driven by the real engine
# --------------------------------------------------------------------------- #
def _flagged_board() -> Board:
    """3x3, mine (1,1) flagged; the left edge is a three-cell column to chord into.

    (1,0) is the probe's best target: value 1, one flagged neighbour, and it opens
    (0,0), (0,1) and (2,1) - three cells, so a chord and a click cannot be confused.
    (2,2) is a second target that shares none of those cells.
    """
    cells = np.array(
        [[COVERED, COVERED, FLAG],
         [1, FLAG, COVERED],
         [COVERED, 1, COVERED]],
        dtype=np.int8,
    )
    return _board(cells, 1)


def _site_screen(accepts: dict[str, bool]) -> tuple[FakeScreen, list[tuple], object]:
    """A 3x3 board and a model of what each gesture form does at the cell it lands on.

    Accepted forms open every neighbour of the target at once - the chord effect.
    A form the site does not accept degrades to what that press really is there: a
    left/middle press and release on an already-revealed number opens nothing, and a
    left+right press that reaches the site as a left release first opens the one cell
    under the cursor (a click), which is exactly what the probe must see and never
    repeat. Like the real site, the state changes only after an animation the probe
    has to wait out: the next read after a gesture is unreadable.
    """
    screen = FakeScreen(_flagged_board().cells, mines_left=1)
    calls: list[tuple] = []

    def neighbours(r, c):
        return [(rr, cc) for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)
                if (rr, cc) != (r, c) and 0 <= rr < 3 and 0 <= cc < 3]

    def apply(mouse) -> None:
        order = [btn for kind, btn, _ in mouse._batch if kind == "down"]
        together = order == ["left", "right"]
        r = (mouse.pos[1] - THEME.margin) // THEME.cell
        c = (mouse.pos[0] - THEME.margin) // THEME.cell
        calls.append((tuple(order) + (together,), (r, c)))
        if order == ["middle"] and accepts.get("middle"):
            opened = [x for x in neighbours(r, c) if screen.cells[x] == COVERED]
        elif order == ["left", "right"] and accepts.get("left+right"):
            opened = [x for x in neighbours(r, c) if screen.cells[x] == COVERED]
        elif order and order[0] == "left":
            opened = [(r, c)] if screen.cells[r, c] == COVERED else []
        else:
            opened = []
        screen.overlay = True      # the animation starts; the next read cannot trust it
        for rr, cc in opened:
            screen.set(rr, cc, 1)

    return screen, calls, apply


@pytest.mark.parametrize("accepts", [{"middle": True, "left+right": True}, {"middle": True}])
def test_probe_prefers_the_middle_button(accepts, fast_clock):
    screen, calls, apply = _site_screen(accepts)
    mouse = SimMouse(apply)
    probe = ex.ChordProbe(screen, ex.Pointer(mouse), between_trials=0.0,
                          settle={"gap": 0.0})
    support = probe.run("fixture")
    assert support.ok and support.method == "middle"
    assert calls[0][0] == ("middle", False)


def test_probe_falls_back_to_left_plus_right_when_middle_is_ignored(fast_clock):
    screen, calls, apply = _site_screen({"left+right": True})
    mouse = SimMouse(apply)
    support = ex.ChordProbe(screen, ex.Pointer(mouse), between_trials=0.0,
                            settle={"gap": 0.0}).run("fixture")
    assert support.method == "left+right"
    assert ("left", "right", True) in [c[0] for c in calls]


def test_no_chording_at_all_is_reported_not_guessed(fast_clock):
    screen, calls, apply = _site_screen({})
    mouse = SimMouse(apply)
    support = ex.ChordProbe(screen, ex.Pointer(mouse), between_trials=0.0,
                            settle={"gap": 0.0}).run("fixture")
    assert not support.ok
    with pytest.raises(ex.ExecError, match="no chord form accepted"):
        support.method
    assert "no chord accepted" in support.summary()


def test_a_click_effect_is_contradicted_by_the_chord_itself():
    """One changed cell is a click; several are a chord; a moved number is a new board."""
    cells = np.full((3, 3), COVERED, np.int8)
    cells[0, 0] = 2
    cells[0, 2] = FLAG
    before = _board(cells, 0)
    trial = ex._Trial((0, 0), ((1, 2), (1, 1)), (0, 0))

    assert ex._effect(before, _board(cells, 0), trial) == "nothing"

    one = cells.copy()
    one[1, 2] = 1
    assert ex._effect(before, _board(one, 0), trial) == "click"

    two = cells.copy()
    two[1, 2] = 1
    two[1, 1] = 0
    assert ex._effect(before, _board(two, 0), trial) == "chord"

    bigger = np.full((6, 6), COVERED, np.int8)
    assert ex._effect(before, _board(bigger, 0), trial) == "reset"


def test_probe_refuses_a_board_with_nothing_to_chord(fast_clock):
    screen = FakeScreen(np.full((4, 4), COVERED, np.int8))
    with pytest.raises(ex.ExecError, match="nothing chordable"):
        ex.ChordProbe(screen, ex.Pointer(SimMouse()), settle={"gap": 0.0}).run("fresh")


def _impossible(moment, r: int, c: int, value: int = COVERED):
    """The settled read, but with cell (r,c) showing something no gesture can produce."""
    forced = moment.board.cells.copy()
    forced[r, c] = value
    return ex.Moment(
        ex.Board(forced.shape[0], forced.shape[1], forced, moment.board.mines_left),
        moment.frames, moment.capture_ms,
    )


def test_the_audit_aborts_on_a_change_no_chord_can_explain():
    """A cell opens where a click on the target cannot have acted: a different board.

    On a 5x5 the pressed number sits at (4,4); a read that opens (1,1) as well is not a
    chord or a flood from (4,4) - it is the "rect no longer holds this board" case.
    """
    cells = np.full((5, 5), COVERED, np.int8)
    cells[4, 4] = 1
    cells[3, 3] = FLAG
    cells[3, 4] = 0                 # opens (4,3) by itself, like a clicked zero would
    before = _board(cells, 1)

    planned = cells.copy()
    planned[4, 3] = 0
    trial = ex._Trial((4, 4), ((4, 3),), (0, 0))
    probe = ex.ChordProbe(FakeScreen(before.cells), ex.Pointer(SimMouse()))
    probe._audit("fixture", "middle", before, _board(planned, 1), trial, set())

    strange = planned.copy()
    strange[1, 1] = 7               # four rows away from the press: nothing there was pressed
    with pytest.raises(ex.ExecError, match="re-probe"):
        probe._audit("fixture", "middle", before, _board(strange, 1), trial, set())


def test_the_audit_ignores_cells_that_already_contradict_their_flags():
    """A board perception already knows is self-contradictory cannot be made worse, and a
    chord next to it must not be thrown out because one impossible cell moved."""
    cells = np.full((3, 2), COVERED, np.int8)
    cells[0, 0] = 0                 # a 0 next to...
    cells[0, 1] = FLAG              # ...a flag: false from the first frame
    cells[1, 0] = 1
    before = _board(cells, 1)
    trial = ex._Trial((1, 0), ((1, 1),), (0, 0))
    after = cells.copy()
    after[1, 1] = 0
    after[2, 1] = 0
    probe = ex.ChordProbe(FakeScreen(before.cells), ex.Pointer(SimMouse()))
    probe._audit("fixture", "middle", before, _board(after, 1), trial, probe._sealed_cells(before))


def test_probe_does_not_abort_on_a_changed_cell_near_the_target(fast_clock):
    """The audit is not a plan-check: a cell next to the target that the plan did not name
    is still a picture the probe can read (one changed cell = click), so it must not abort.

    This is the check that keeps the audit honest in the other direction: a probe that
    aborted on every unexpected-within-reach change would be unable to read the very form
    (left+right pressed late) it exists to detect.
    """
    screen, calls, apply = _site_screen({"middle": True})
    real_wait = ex.wait_stable
    calls_made = {"n": 0}

    def wait_then_mark(*args, **kwargs):
        moment = real_wait(*args, **kwargs)
        calls_made["n"] += 1
        if calls_made["n"] == 2:
            forced = moment.board.cells.copy()
            forced[1, 2] = 4        # a neighbour of (1,0), not in the plan for it
            moment = ex.Moment(
                ex.Board(forced.shape[0], forced.shape[1], forced, moment.board.mines_left),
                moment.frames, moment.capture_ms,
            )
        return moment

    ex.wait_stable = wait_then_mark
    try:
        support = ex.ChordProbe(screen, ex.Pointer(SimMouse(apply)), between_trials=0.0,
                                settle={"gap": 0.0}).run("fixture")
    finally:
        ex.wait_stable = real_wait
    assert support.ok, support.details


def test_probe_calls_the_guard_before_every_gesture(fast_clock):
    screen, calls, apply = _site_screen({"middle": True})
    seen: list[int] = []
    ex.ChordProbe(screen, ex.Pointer(SimMouse(apply)), guard=lambda: seen.append(1),
                  between_trials=0.0, settle={"gap": 0.0}).run("fixture")
    assert seen


def test_trials_skip_candidates_next_to_unreadable_cells():
    from ms.board import UNREADABLE

    cells = _flagged_board().cells
    cells[0, 1] = UNREADABLE
    board = _board(cells, 0)
    targets = [t.target for t in ex._trials(board, lambda cell: (0, 0))]
    assert (0, 0) not in targets
    assert chord_candidates(board)


# --------------------------------------------------------------------------- #
# through real perception: the pixels actually decide
# --------------------------------------------------------------------------- #
def _perception_on(cells: np.ndarray) -> Perception:
    """A reader calibrated on an all-covered frame of the given shape, like the agent's."""
    rows, cols = cells.shape
    m, c = THEME.margin, THEME.cell
    perc = Perception(Rect(m, m, cols * c, rows * c), rows, cols)
    perc.calibrate(_render(np.full((rows, cols), COVERED, np.int8)))
    return perc


def test_two_rendered_frames_of_one_board_settle(fast_clock):
    cells = np.full((6, 6), COVERED, np.int8)
    cells[2, 2] = 1
    cells[2, 3] = 2
    img = _render(cells)
    perc = _perception_on(cells)

    class P:
        reads = 0

        def capture(self):
            P.reads += 1
            return img.copy()

        def board_at(self, frame):
            return perc.read(frame, 4)

    moment = ex.wait_stable(P(), gap=0.05)
    assert moment.board.rows == 6 and moment.board.mines_left == 4
    assert int(moment.board.cells[2, 2]) == 1
    assert sorted(int(v) for v in np.unique(moment.board.cells)) == [COVERED, 1, 2]


def test_a_settle_waits_for_the_gap_between_its_two_reads(fast_clock):
    """A still board settles on its second read - after the gap has actually passed.

    The loop's own accounting is what is checked here: two reads, and at least `gap` of
    simulated time between them. Settling earlier is the failure mode (an animation can
    hold one frame for a whole animation step), so the gap is a contract, not a retry.
    """
    cells = np.full((4, 4), COVERED, np.int8)
    s = FakeScreen(cells)
    ex.wait_stable(s, gap=0.05, poll=0.01)
    assert s.captures == 2, "a still board needs exactly two reads"
    assert fast_clock.sleeps and sum(fast_clock.sleeps) >= 0.05
    assert s.reads == 2


def test_a_board_mid_animation_does_not_settle_early(fast_clock):
    """What the loop returns are the two reads the gap actually separates.

    The second capture is taken `poll` after the first, which is too early to be evidence.
    So the pair it hands back is (first read, third read): two captures `gap` apart, of
    the same board. That is the whole contract — not "two reads in a row".
    """
    cells = np.full((4, 4), COVERED, np.int8)
    still = _render(cells)
    perc = _perception_on(cells)

    class Steady:
        n = 0

        def capture(self):
            Steady.n += 1
            return still.copy()

        def board_at(self, frame):
            return perc.read(frame, 4)

    moment = ex.wait_stable(Steady(), gap=0.05, poll=0.01)
    assert fast_clock.sleeps, "the gap must be waited out, not assumed"
    assert sum(fast_clock.sleeps) >= 0.05
    assert np.array_equal(moment.frames[0], still) and np.array_equal(moment.frames[1], still)
