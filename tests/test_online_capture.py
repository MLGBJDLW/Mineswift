from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from ms.desktop_grid import DesktopReader, find_lattice
from ms.play_web import visible_rows
from ms.web_vision import GridLayout


OCEAN_ROWS = [
    "#########1001###", "#########10012##", "#########1000111",
    "#########1000000", "#########3121100", "#############100",
    "#############111", *(["################"] * 9),
]


@pytest.mark.parametrize("covered", [(0, 0), None])
@pytest.mark.parametrize("scale", [0.5, 0.75, 1, 1.25])
def test_resume_flat_coloured_board(covered, scale):
    from ms.perception import Calibration

    with Image.open(Path(__file__).with_name("fixtures") / "ocean-intermediate.png") as source:
        image = source.resize((round(source.width * scale), round(source.height * scale)),
                              Image.Resampling.BILINEAR)
        box = find_lattice(np.asarray(image), 16, 16)
        pixels = np.asarray(image.crop(box))
    height, width = pixels.shape[:2]
    reader = DesktopReader(pixels, (16, 16), covered, 40, classic_colors=True,
                           calibration=Calibration(16, 16, 0, 0, width / 16, height / 16))
    board = reader.decode(pixels, GridLayout(box, 16, 16)).board
    assert visible_rows(board) == OCEAN_ROWS


def test_bevel_and_gentle_gradient_leave_a_blank_face_readable():
    from ms.tile_vision import observe

    ramp = np.linspace(-5, 5, 28).round().astype(int)[:, None, None]
    face = np.full((28, 28, 3), (178, 155, 137), dtype=int) + ramp
    face[[0, -1], :, :] -= 6
    face[:, [0, -1], :] -= 6
    assert observe(face.astype(np.uint8))[1] is None
    face[6:21, 14:16] -= 7
    assert observe(face.astype(np.uint8))[1] is not None


def test_right_click_confirms_a_rectangular_flag_and_can_reattach():
    from ms.board import FLAG
    from ms.perception import Calibration
    from ms.play_web import verify_change
    with Image.open(Path(__file__).with_name("fixtures") / "classic-stalled.png") as im:
        base = np.array(im.convert("RGB").resize((297,297), Image.Resampling.NEAREST))
    p = 33
    cell = (6, 4)
    tile = base[6*p:7*p, 4*p:5*p]
    tile[:] = base[:p, :p]
    layout = GridLayout((0, 0, 297, 297), 9, 9)
    def reader(pixels):
        return DesktopReader(pixels, (9,9), (0,0), 10, calibration=Calibration(9,9,0,0,p,p))
    r = reader(base)
    before = r.decode(base, layout)
    marked = base.copy()
    tile = marked[6*p:7*p, 4*p:5*p]
    ink = [70, 83, 96]
    tile[8:25, 9:11] = ink
    for x in range(9, 23):
        offset = (x-9)//6
        tile[8+offset:18+offset, x] = ink
    after = r.decode(marked, layout, (base, before.board.cells))
    verify_change(before, after, cell, "flag")
    assert reader(marked).decode(marked, layout).board.cells[cell] == FLAG


def coloured_grid(seed, equal_luminance):
    import cv2

    rng = np.random.default_rng(seed)
    colours = []
    while len(colours) < 13:
        colour = rng.integers(12, 245, 3)
        if equal_luminance:
            colour[1] = round((126 - .299 * colour[0] - .114 * colour[2]) / .587)
        if 0 <= colour[1] <= 255 and all(np.linalg.norm(colour - c) > 52 for c in colours):
            colours.append(colour)
    grid, cover1, cover2, open1, open2, *inks = colours
    cells = np.full((9, 9), -1, np.int8)
    cells[2:6, 2:7] = [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [8, 7, 6, 5, 0], [0, 0, 0, 1, 2]]
    pitch, margin = 40, 13
    image = np.full((9*pitch + 2*margin, 9*pitch + 2*margin, 3), grid, np.uint8)
    for (row, col), value in np.ndenumerate(cells):
        x, y = margin + col*pitch, margin + row*pitch
        faces = (cover1, cover2) if value < 0 else (open1, open2)
        image[y+1:y+pitch-1, x+1:x+pitch-1] = faces[(row + col) % 2]
        if value > 0:
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            (w, h), _ = cv2.getTextSize(str(value), font, 1.3, 2)
            cv2.putText(image, str(value), (x+(pitch-w)//2, y+(pitch+h)//2),
                        font, 1.3, inks[value-1].tolist(), 2, cv2.LINE_AA)
    return image, cells


@pytest.mark.parametrize("seed", [27, 91, 831, 1001])
@pytest.mark.parametrize("equal_luminance", [True, False])
def test_full_window_random_checkerboard_palettes(seed, equal_luminance):
    from ms.perception import Calibration

    image, expected = coloured_grid(seed, equal_luminance)
    box = find_lattice(image, 9, 9)
    assert max(abs(a-b) for a, b in zip(box, (13, 13, 373, 373))) <= 1
    x, y, r, b = box
    pixels = image[y:b, x:r]
    reader = DesktopReader(pixels, (9, 9), None, 10, classic_colors=True,
                           calibration=Calibration(9, 9, 0, 0, (r-x)/9, (b-y)/9))
    frame = reader.decode(pixels, GridLayout(box, 9, 9))
    np.testing.assert_array_equal(frame.board.cells, expected)


def test_recoloured_dark_board_and_faint_mark_stop():
    from PIL import ImageOps
    from ms.board import Unreadable
    from ms.perception import Calibration

    with Image.open(Path(__file__).with_name("fixtures") / "ocean-intermediate.png") as source:
        image = ImageOps.invert(source)
    box = find_lattice(np.asarray(image), 16, 16)
    pixels = np.array(image.crop(box))
    h, w = pixels.shape[:2]
    reader = DesktopReader(pixels, (16, 16), None, 40,
                           calibration=Calibration(16, 16, 0, 0, w/16, h/16))
    layout = GridLayout(box, 16, 16)
    frame = reader.decode(pixels, layout)
    assert visible_rows(frame.board) == OCEAN_ROWS
    changed = pixels.copy()
    tile = reader.crop(changed, 0, 10)
    tile[12:25, 15:20] = tile[10, 10] + 4
    with pytest.raises(Unreadable, match="无法确认"):
        reader.decode(changed, layout, previous=(pixels, frame.board.cells))


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


def test_last_reveal_verifies_new_tiles_under_banner():
    from ms.desktop_grid import DesktopGame
    from ms.perception import Calibration
    from ms.board import COVERED, FLAG

    with Image.open(Path(__file__).with_name("fixtures") / "classic-stalled.png") as im:
        sheet = np.array(im)
    tiles = {value: sheet[r*24:r*24+24, c*24:c*24+24]
             for value, (r,c) in {0:(0,4), 1:(0,3), COVERED:(0,0), FLAG:(6,4)}.items()}
    cells = np.zeros((9,9), np.int8)
    cells[:2,:2] = 1
    cells[0,0] = cells[-1,-1] = COVERED
    start = np.vstack([np.hstack([tiles[v] for v in row]) for row in cells])
    end = start.copy()
    end[:24,:24] = tiles[FLAG]
    end[-24:,-24:] = tiles[0]
    end[24:48, 24:55] = [200, 10, 145]  # a banner obscures a previously read clue
    port = DesktopGame.__new__(DesktopGame)
    port.layout, port.title = GridLayout((0,0,216,216), 9, 9), "Local visual test"
    port.reader = DesktopReader(start, (9,9), (0,0), 1, classic_colors=True,
                                calibration=Calibration(9,9,0,0,24,24))
    port.capture = lambda: start
    before = port.read()
    port.press = lambda *args: None
    port.click(before, (8,8), "left")
    assert port.final_reveal(end).status == "won"
    assert port.final_reveal(start) is None
    end[-24:,-24:] = tiles[1]  # the new clue contradicts its neighbours
    assert port.final_reveal(end) is None


def test_custom_size_grid_from_lines_without_page_metadata():
    from ms.desktop_grid import locate_lattices
    rng = np.random.default_rng(123)
    rows, cols, pitch = 12, 18, 19
    image = np.full((280, 440, 3), 240, np.uint8)
    field = image[21:21+rows*pitch+1, 28:28+cols*pitch+1]
    field[:] = rng.integers(125, 159, field.shape, dtype=np.uint8)
    field[::pitch, :] = 16
    field[:, ::pitch] = 16
    matches = locate_lattices(image)
    assert len(matches) == 1 and matches[0][0] == (rows, cols)
    assert max(abs(a-b) for a,b in zip(matches[0][1], (28,21,28+cols*pitch,21+rows*pitch))) <= 2


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
