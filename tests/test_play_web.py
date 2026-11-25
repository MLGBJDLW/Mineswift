import base64
import io
import sys

import numpy as np
import pytest

pytest.importorskip("PIL")
from PIL import Image

from ms.board import Board, COVERED, FLAG, Stuck, Unreadable
from ms.engine import INTERMEDIATE, Minefield
from ms.executor import WindowsMouse
from ms.play_web import Controls, best_chord, load_policy, play, run_rounds, verify_change
from ms.web_chrome import ChromeGame, RestartRound, settled
from ms.web_vision import Frame, Layout, Sprites, decode


@pytest.fixture(scope="module")
def artwork():
    # Self-contained synthetic artwork, including all public rendering states.
    rng = np.random.default_rng(7)
    images, records = {}, []
    for name, count, size in (("bgf", 9, (25, 25)), ("sgf", 4, (25, 25)),
                              ("dgf", 10, (13, 23)), ("fgf", 3, (21, 21))):
        for i in range(count):
            color = rng.integers(0, 256, (4, 4, 3), dtype=np.uint8)
            image = Image.fromarray(color).resize(size, Image.Resampling.NEAREST)
            output = io.BytesIO()
            image.save(output, format="GIF")
            records.append(f'{name}[{i}]="data:img/gif;base64,'.encode()
                           + base64.b64encode(output.getvalue()) + b'"')
            images[name, i] = image
    return images, Sprites(b"\n".join(records))


@pytest.mark.parametrize("scale", [1, 1.25, 1.5, 2])
def test_visible_board_at_multiple_scales(artwork, scale):
    images, sprites = artwork
    cells = [[0, 1, 2, 3, 4], [5, 6, 7, 8, FLAG], [COVERED]*5]
    layout = Layout((0, 33, 129, 112), (54, 0, 79, 25), (0, 1, 39, 24))
    image = Image.new("RGB", (129, 112), "gray")
    image.paste(images["fgf", 0], (56, 2))
    for i, digit in enumerate([0, 0, 3]):
        image.paste(images["dgf", digit], (13*i, 1))
    for r, row in enumerate(cells):
        for c, val in enumerate(row):
            key = ("bgf", val) if val >= 0 else ("sgf", int(val == FLAG))
            image.paste(images[key], (2 + 25*c, 35 + 25*r))
    scaled = Layout(*(tuple(round(v*scale) for v in box) for box in (layout.grid, layout.face, layout.counter)))
    screen = image.resize((round(129*scale), round(112*scale)), Image.Resampling.BILINEAR)
    frame = decode(screen, scaled, sprites)
    assert frame.status == "playing" and frame.board.mines_left == 3
    np.testing.assert_array_equal(frame.board.cells, cells)
    x, y = scaled.center((2, 3))
    screen.paste("white", (x-round(11*scale), y-round(11*scale), x+round(11*scale), y+round(11*scale)))
    with pytest.raises(Unreadable):
        decode(screen, scaled, sprites)


def position(cells, remaining):
    grid = np.asarray(cells, dtype=np.int8)
    layout = Layout((0, 25, 25*grid.shape[1]+4, 25*grid.shape[0]+29), (0, 0, 25, 25), (26, 0, 65, 23))
    return Frame(layout, "playing", Board(*grid.shape, grid, remaining))


def test_expert_dimensions_with_outward_rounded_accessibility_bounds():
    # A 30 by 16 board at 75%, with a 17.25-pixel-high counter reported as 18.
    layout = Layout((10, 40, 576, 343), (280, 10, 299, 29), (15, 10, 45, 28))
    assert layout.geometry()[:2] == (16, 30)


@pytest.mark.parametrize("index,status", [(1, "won"), (2, "lost")])
def test_terminal_face_ends_game_without_clicking(artwork, index, status):
    images, sprites = artwork
    layout = position([[COVERED]*3], 1).layout
    image = Image.new("RGB", (79, 54), "white")
    image.paste(images["fgf", index], (2, 2))
    frame = decode(image, layout, sprites)
    assert play(None, None, frame, 1, lambda *a, **kw: None) == (status, {})


def test_web_midgame_loads_actor_for_a_genuine_guess():
    pytest.importorskip("torch")
    frame = position([[1, 1], [COVERED, COVERED]], 1)
    policy = load_policy(71)
    seen = []
    policy.actor.register_forward_pre_hook(lambda _, args: seen.append(args[0].numpy().copy()))
    events = []

    class TwoChoices:
        def click(self, before, cell, button):
            assert before.matches(frame) and cell in ((1, 0), (1, 1)) and button == "left"
            self.result = Frame(frame.layout, "lost" if cell == (1, 0) else "won")

        def read(self):
            return self.result

    result, counts = play(TwoChoices(), policy, frame, 71, lambda event, **v: events.append((event, v)))
    assert result in ("won", "lost") and counts == {"model": 1}
    assert policy.guesses == 1 and len(seen) == 1
    np.testing.assert_array_equal(seen[0][0], frame.board.cells)
    assert events[0][1]["source"] == "model"


def test_random_opening_with_an_existing_flag_skips_the_flag():
    frame = position([[FLAG, COVERED, COVERED]], 0)

    class Unopened:
        def click(self, before, cell, button):
            assert cell in ((0, 1), (0, 2)) and button == "left"

        def read(self):
            return Frame(frame.layout, "won")

    _, counts = play(Unopened(), None, frame, 2, lambda *a, **kw: None)
    assert counts == {"opening": 1}


def test_resumed_game_validates_a_preexisting_wrong_flag():
    pytest.importorskip("torch")
    frame = position([[FLAG, 0, 1, COVERED]], 0)
    uncovered_flag = position([[COVERED, 0, 1, COVERED]], 1)
    clicks = []

    class AlreadyStarted:
        def click(self, before, cell, button):
            clicks.append((cell, button))
            self.next = uncovered_flag if len(clicks) == 1 else Frame(frame.layout, "won")

        def read(self):
            return self.next

    _, counts = play(AlreadyStarted(), load_policy(1), frame, 1, lambda *a, **kw: None)
    assert clicks == [((0, 0), "right"), ((0, 0), "left")]
    assert counts == {"solver": 1, "unflag": 1}


def test_swallowed_or_unexpected_click_stops():
    before = position([[1, COVERED, COVERED]], 1)
    class Still:
        def read(self):
            return before
    with pytest.raises(Stuck):
        settled(Still(), before, timeout=.1)
    with pytest.raises(Stuck):
        verify_change(before, position([[1, FLAG, FLAG]], 0), (0, 1), "flag")


def test_one_retry_for_a_swallowed_open():
    before = position([[COVERED, COVERED]], 1)
    class Delayed:
        clicks = 0
        def click(self, frame, cell, button):
            assert frame.matches(before) and button == "left"
            self.clicks += 1
        def read(self):
            return Frame(before.layout, "won") if self.clicks == 2 else before
    port = Delayed()
    events = []
    play(port, None, before, 1, lambda event, **data: events.append(event),
         read_after=lambda io, **kwargs: io.read())
    assert port.clicks == 2 and events.count("input_retry") == 1


def test_unreadable_frame_disables_unchanged_retry(monkeypatch):
    from ms import web_chrome
    from ms.board import Unreadable

    frame = position([[1, COVERED, COVERED]], 1)
    times = iter(n / 20 for n in range(10))
    monkeypatch.setattr(web_chrome.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(web_chrome.time, "sleep", lambda _: None)
    class Flicker:
        first = True
        def read(self):
            if self.first:
                self.first = False
                raise Unreadable("动画帧")
            return frame
    with pytest.raises(web_chrome.ReadbackError, match="动画帧"):
        settled(Flicker(), frame, timeout=.25, allow_unchanged=True)


def test_native_mouse_up_flags_for_single_click_and_cleanup():
    mouse = WindowsMouse.__new__(WindowsMouse)
    mouse._down = set()
    sent = []
    mouse._send = sent.extend
    mouse.press("right")
    mouse.release("right")
    mouse.pressed_together(("left", "middle"))
    mouse.release_all()
    assert [flag for flag, _ in sent] == [8, 16, 2, 32, 4, 64]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows session mutex")
def test_simultaneous_players_cannot_click_the_same_game():
    port = ChromeGame.__new__(ChromeGame)
    with port.one_player(), pytest.raises(Stuck), port.one_player():
        pytest.fail("second player acquired the mouse")
    with port.one_player():
        pass


def test_chord_requires_a_proved_flag_and_all_safe_targets():
    f = position([[FLAG,1,COVERED], [1,1,COVERED]], 0)
    safe = [(0,2),(1,2)]
    anchor, opens = best_chord(f.board, f.board, safe)
    assert set(opens) == set(safe)
    logical = position([[COVERED,1,COVERED],[1,1,COVERED]], 1)
    assert best_chord(logical.board, f.board, safe) is None
    with pytest.raises(Stuck):
        verify_change(f, position([[FLAG,1,1],[1,1,COVERED]],0), anchor, "chord", opens)
    with pytest.raises(Stuck):
        verify_change(f, Frame(f.layout, "lost"), anchor, "chord", opens)


class SimulatedPort:
    def __init__(self, seed):
        self.field = Minefield(INTERMEDIATE, np.random.default_rng(seed * 1_000_003 + 7))
        self.layout = Layout((0, 30, 404, 434), (100, 0, 125, 25), (0, 0, 39, 23))

    def click(self, frame, cell, button):
        row, col = cell
        index = row * self.field.cfg.cols + col
        actions = {"left": self.field.click, "right": self.field.flag, "chord": self.field.chord}
        actions[button](index)

    def read(self):
        if self.field.won:
            return Frame(self.layout, "won")
        if self.field.exploded:
            return Frame(self.layout, "lost")
        return Frame(self.layout, "playing", self.field.board())


def test_blind_and_chord_on_identical_layouts():
    pytest.importorskip("torch")
    for seed in (660000, 660001):
        results = []
        for mode in ("single", "chord", "blind"):
            port, events = SimulatedPort(seed), []
            def emit(event, **data):
                events.append((event, data))
            outcome, counts = play(port, load_policy(seed), port.read(), seed, emit,
                                   chords=mode=="chord", blind=mode=="blind",
                                   read_after=lambda io, **kw: io.read())
            results.append(outcome)
            if mode == "chord":
                assert counts["chords"] > 0
            if mode == "blind":
                assert any(e == "remembered" for e,d in events)
                assert any(d.get("memory") for _,d in events)
                assert all(d.get("action") not in ("chord","flag") for _,d in events)
                assert not np.any(port.field.visible == FLAG)
        assert len(set(results)) == 1


def test_noop_chord_falls_back_to_single_clicks():
    pytest.importorskip("torch")
    port = SimulatedPort(660000)
    click = port.click
    attempts = []
    def no_chord(frame, cell, button):
        attempts.append(button)
        if button != "chord":
            click(frame,cell,button)
    port.click = no_chord
    play(port,load_policy(1),port.read(),660000,lambda *a,**k: None,
         read_after=lambda io, **kw: io.read())
    assert attempts.count("chord") == 1 and "left" in attempts[attempts.index("chord")+1:]


def test_session_restarts_terminal_once_and_resets_game_statistics(monkeypatch):
    import ms.play_web as player
    f = position([[COVERED, COVERED]], 1)
    class Port:
        restarts = 0
        def guard(self):
            raise RestartRound()
        def restart(self):
            self.restarts += 1
            return f
    port, logs = Port(), []
    monkeypatch.setattr(player, "load_policy", lambda seed: type("Policy", (), {"name":"test"})())
    monkeypatch.setattr(player, "play", lambda *a,**kw: ("won", {}))
    run_rounds(port, Frame(f.layout,"lost"), 3, lambda event,**d: logs.append((event,d)),
               controls=Controls(),games=1)
    assert port.restarts == 1
    assert [d["seed"] for e,d in logs if e=="attached"] == [3,4]


def test_r_is_edge_triggered_scoped_and_ignores_edit_fields():
    from types import SimpleNamespace
    g = ChromeGame.__new__(ChromeGame)
    state = {ord("R"): 0x8001}
    g.hwnd, g.controls, g.restarting, g.r_down = 3, Controls(), False, False
    g.mouse = SimpleNamespace(foreground_window=lambda:3,user32=SimpleNamespace(GetAsyncKeyState=lambda k:state.get(k,0)))
    focused = SimpleNamespace(CurrentControlType=5)
    g.ui = SimpleNamespace(GetFocusedElement=lambda:focused)
    g.u = SimpleNamespace(UIA_EditControlTypeId=7)
    g._host_ok = lambda: True
    with pytest.raises(RestartRound):
        g.guard()
    g.guard()
    state.clear()
    g.guard()
    focused.CurrentControlType = 7
    state[ord("R")] = 0x8001
    g.guard()
    g._host_ok = lambda: False
    with pytest.raises(Stuck):
        g.guard()


def test_monitor_is_outside_board_and_guess_rounding_preserves_uncertainty():
    pytest.importorskip("tkinter")
    from ms.web_monitor import placement, risk_text
    for bounds, work in [((300,100,750,700),(0,0,1280,800)),
                         ((-900,10,-300,580),(-1280,0,0,800))]:
        w,h,x,y = placement(bounds,work)
        assert x+w <= bounds[0] or x >= bounds[2] or y+h <= bounds[1] or y >= bounds[3]
    assert placement((0,0,990,890),(0,0,1000,900)) is None
    assert [risk_text(x) for x in (.00001,.5,.9999)] == ["<0.1%","50.0%",">99.9%"]
