from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from ms.desktop_grid import DesktopReader, find_lattice
from ms.play_web import visible_rows
from ms.web_vision import GridLayout


@pytest.mark.parametrize("pitch", [16, 24, 36])
def test_readback_after_opening_red_three(pitch):
    from ms.perception import Calibration

    with Image.open(Path(__file__).with_name("fixtures") / "classic-stalled.png") as source:
        pixels = np.asarray(source.resize((9*pitch, 9*pitch)))
    reader = DesktopReader(pixels, (9, 9), None, 10, classic_colors=True,
                           calibration=Calibration(9, 9, 0, 0, pitch, pitch))
    board = reader.decode(pixels, GridLayout((0, 0, 9*pitch, 9*pitch), 9, 9)).board
    assert board.mines_left == 9
    assert visible_rows(board) == [
        "##2100000", "#3#100000", "121101110", "000001#21", "0000012##",
        "0001111##", "0012F21##", "001######", "001######",
    ]


@pytest.mark.parametrize("scale", [1, 2 / 3, 1.25])
def test_resume_online_hd_from_a_window_screenshot(scale):
    pytest.importorskip("cv2")
    with Image.open(
        Path(__file__).with_name("fixtures") / "online-beginner.png"
    ) as source:
        image = source.resize(
            (round(source.width * scale), round(source.height * scale))
        )
    box = find_lattice(np.asarray(image), 9, 9)
    assert np.max(np.abs(np.array(box) - np.array([28, 124, 352, 448]) * scale)) < 3
    pixels = np.asarray(image.crop(box))
    reader = DesktopReader(pixels, (9, 9), None, 10, classic_colors=True)
    board = reader.decode(pixels, GridLayout(box, 9, 9)).board
    assert visible_rows(board) == [
        "000000000",
        "110000011",
        "#2221001#",
        "####10011",
        "####10000",
        "####10000",
        "####20000",
        "####20000",
        "####10000",
    ]


def test_hybrid_pays_for_a_flag_only_when_the_chord_saves_gestures():
    from ms.board import Board
    from ms.play_web import economical_chord

    def board(cells, left):
        return Board(2, 3, np.array(cells, dtype=np.int8), left)

    logical = board([[-2, 1, -1], [-1, -1, -1]], 0)
    screen = board([[-1, 1, -1], [-1, -1, -1]], 1)
    assert economical_chord(logical, screen, [(0, 2)]) == ("flag", (0, 0), ())
    assert economical_chord(logical, logical, [(0, 2)])[:2] == ("chord", (0, 1))
    screen.cells[1, 2] = -2  # inherited flag has not been proved
    assert economical_chord(logical, screen, [(0, 2)]) is None


def test_online_face_is_centred_above_the_grid():
    from ms.desktop_grid import online_face

    with Image.open(Path(__file__).with_name("fixtures") / "online-beginner.png") as im:
        x, y = online_face(np.asarray(im), (28, 124, 352, 448), 9)
    assert abs(x - 190) <= 2 and abs(y - 62) <= 3


def test_chord_with_equal_gesture_count_can_save_a_slow_reveal():
    from ms.board import Board
    from ms.play_web import economical_chord

    logical = Board(2, 2, np.array([[-2, 1], [-1, -1]], dtype=np.int8), 0)
    physical = Board(2, 2, np.array([[-1, 1], [-1, -1]], dtype=np.int8), 1)
    safe = [(1, 0), (1, 1)]
    assert economical_chord(logical, physical, safe) is None
    assert economical_chord(logical, physical, safe, .13, 1.04) == ("flag", (0, 0), ())
    assert economical_chord(logical, physical, safe, .13, 2.2) is None


def test_physical_mouse_moves_only_to_requested_cells(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from ms import desktop_grid

    moves, buttons = [], []
    mouse = SimpleNamespace(user32=None, move=lambda *xy: moves.append(xy),
                            cursor=lambda: moves[-1], grabbed=nullcontext,
                            press=buttons.append, release=buttons.append,
                            pressed_together=buttons.append, released_together=buttons.append)
    port = desktop_grid.DesktopGame.__new__(desktop_grid.DesktopGame)
    port.mouse, port.hwnd, port.region, port.guard = mouse, 17, (0, 0, 100, 100), lambda: None
    monkeypatch.setattr(desktop_grid, "_window_at", lambda user, x, y: port.hwnd)
    monkeypatch.setattr(desktop_grid.time, "sleep", lambda _: None)
    targets = [(20, 15), (32, 45), (65, 73)]
    for target, button in zip(targets, ("left", "right", "chord")):
        port.press(target, button)
        assert mouse.cursor() == target
    assert moves == targets
    assert buttons == ["left", "left", "right", "right", ("left", "right"), ("right", "left")]


def test_live_read_recaptures_every_time_and_classifies_only_changed_pixels(monkeypatch):
    from unittest.mock import Mock
    from ms import desktop_grid
    from ms.board import FLAG
    from ms.perception import Calibration

    with Image.open(Path(__file__).with_name("fixtures") / "classic-stalled.png") as source:
        screen = np.array(source)
    port = desktop_grid.DesktopGame.__new__(desktop_grid.DesktopGame)
    port.layout = GridLayout((0, 0, 216, 216), 9, 9)
    port.title = "Minesweeper Online"
    port.reader = DesktopReader(screen, (9, 9), None, 10, classic_colors=True,
                                calibration=Calibration(9, 9, 0, 0, 24, 24))
    port.capture = Mock(return_value=screen)
    classify = Mock(wraps=desktop_grid._tile)
    monkeypatch.setattr(desktop_grid, "_tile", classify)
    first = port.read()
    classify.reset_mock()
    assert port.read().matches(first)
    assert classify.call_count == 0
    screen[:24, :24] = screen[144:168, 96:120]
    changed = port.read()
    assert changed.board.cells[0, 0] == FLAG and changed.board.mines_left == 8
    assert classify.call_count == 1 and port.capture.call_count == 3


def test_classic_score_prompt_requires_a_game_specific_message():
    from ms.desktop_grid import _score_message
    assert _score_message("36 is a new all-time high score! Please enter your name")
    assert _score_message("Please enter your name to submit your score (36)")
    assert not _score_message("Please enter your name")


def test_window_binding_uses_the_detected_grid_without_fitting_the_crop_again(monkeypatch):
    from contextlib import ExitStack
    from queue import Queue
    from types import SimpleNamespace
    from ms import app_session
    from ms.play_web import Controls

    with Image.open(Path(__file__).with_name("fixtures") / "online-beginner.png") as im:
        screenshot = im.copy()
    title = "Game #1 - Minesweeper Online - Google Chrome"
    monkeypatch.setattr(app_session, "windows", lambda: {123: title})
    monkeypatch.setattr(app_session, "window_capture", lambda _: ((0, 0, *screenshot.size), title, screenshot))
    monkeypatch.setattr(DesktopReader, "fit", lambda *args: pytest.fail("duplicated grid fit"))
    monkeypatch.setattr(app_session.DesktopGame, "calibrated",
                        lambda hwnd, rect, title, box, reader, restart: SimpleNamespace(
                            layout=GridLayout(box, reader.rows, reader.cols), name="desktop"))
    session = app_session.Session(Queue(), Queue(), Controls())
    with ExitStack() as stack:
        session.bind(stack, dict(hwnd=123, url=None, shape=None, mines=None, rules="classic", classic_colors=True))
    assert session.events.get_nowait()[0] == "capture_image"
    name, preview = session.events.get_nowait()
    assert name == "preview" and preview["board"][2] == "#2221001#"
