import numpy as np
import pytest

from ms.board import Board, COVERED, FLAG, Stuck, Unreadable
from ms.solver import ORTHOGONAL, Solver


def test_four_neighbours_exclude_diagonal_and_raw_policy_handles_a_guess():
    from ms.play_web import load_policy

    board = Board(2, 2, np.array([[1, COVERED], [COVERED, FLAG]], dtype=np.int8), 0)
    classic = Solver().deduce(board)
    assert classic.kind == "open" and len(classic.cells) == 2
    from ms.solver import Inconsistent

    # Orthogonal 1 still requires a mine, while the remaining global budget is zero.
    with pytest.raises(Inconsistent):
        Solver(max_nodes=None, offsets=ORTHOGONAL).deduce(board)
    ambiguous = Board(
        2, 2, np.array([[1, COVERED], [COVERED, COVERED]], dtype=np.int8), 1
    )
    move = Solver(max_nodes=None, offsets=ORTHOGONAL).deduce(ambiguous)
    assert move.kind == "open" and move.cells == ((1, 1),)
    pytest.importorskip("torch")
    policy = load_policy(3, "orthogonal")
    guess = Board(1, 3, np.array([[COVERED, 1, COVERED]], dtype=np.int8), 1)
    assert policy.act(guess).kind == "guess" and policy.guesses == 1


def test_desktop_theme_samples_and_geometry(tmp_path):
    pytest.importorskip("cv2")
    from PIL import Image
    from ms.desktop_grid import DesktopReader
    from ms.web_vision import GridLayout
    from tests._synthetic import render, random_theme

    theme = random_theme(np.random.default_rng(12), cell=24)

    def draw(cells):
        image, (x, y) = render(cells, theme)
        return image[y : y + 6 * 24, x : x + 6 * 24]

    cells = np.full((6, 6), COVERED, dtype=np.int8)
    fresh = draw(cells)
    layout = GridLayout((0, 0, 144, 144), 6, 6)
    reader = DesktopReader(fresh, (6, 6), (0, 0), 5)
    assert reader.decode(fresh, layout).board.covered().all()
    cells[2, :] = [0, 1, 2, 3, 4, FLAG]
    partial = draw(cells)
    reader.save_samples(
        partial, tmp_path, [(str(i), 3, i + 1) for i in range(5)] + [("flag", 3, 6)]
    )
    reader = DesktopReader(fresh, (6, 6), (0, 0), 5, tmp_path)
    frame = reader.decode(partial, layout)
    np.testing.assert_array_equal(frame.board.cells, cells)
    assert frame.board.mines_left == 4
    with pytest.raises(ValueError):
        reader.save_samples(partial, tmp_path, [("flag", 3, 6)])
    covered_tile = reader.crop(fresh, 1, 1)
    hit = np.full_like(covered_tile, [255, 25, 35])
    Image.fromarray(hit).save(tmp_path / "hit.png")
    reader = DesktopReader(fresh, (6, 6), (0, 0), 5, tmp_path)
    image = fresh.copy()
    image[24:48, 24:48] = hit
    assert reader.decode(image, layout).status == "lost"


def test_desktop_unknown_or_blank_image_stops():
    pytest.importorskip("cv2")
    from ms.desktop_grid import DesktopReader

    with pytest.raises(Unreadable):
        DesktopReader(np.zeros((120, 120, 3), np.uint8), (6, 6), (0, 0), 4)


def test_activation_waits_until_the_game_has_focus(monkeypatch):
    from unittest.mock import Mock
    from ms import app_session

    api = Mock()
    api.IsIconic.return_value = False
    api.GetForegroundWindow.side_effect = [20, 0, 31]
    monkeypatch.setattr(app_session.ctypes, "windll", Mock(user32=api), raising=False)
    sleep = Mock()
    monkeypatch.setattr(app_session.time, "sleep", sleep)
    app_session.activate(31)
    api.SetForegroundWindow.assert_called_once_with(31)
    sleep.assert_called_once()


def test_opencv_window_detection_and_rejection_of_two_boards():
    pytest.importorskip("cv2")
    from ms.desktop_grid import find_lattice
    window=np.full((210,460,3),20,np.uint8)
    def draw(x,y):
        for row,col in np.ndindex((6,8)):
            l,t=x+col*20,y+row*20
            window[t:t+20,l:l+20]=180
            window[t+2:t+18,l+2:l+18]=220
    draw(25,40)
    assert find_lattice(window,6,8)==(25,40,185,160)
    draw(260,40)
    with pytest.raises(Unreadable):
        find_lattice(window,6,8)


def test_unknown_grid_size_uses_the_visible_period():
    from ms.perception import fit_grid,Rect
    from tests._synthetic import render,random_theme
    cells=np.full((9,12),COVERED,np.int8)
    img,(x,y)=render(cells,random_theme(np.random.default_rng(12),cell=24))
    c=fit_grid(img,Rect(x,y,12*24,9*24))
    assert (c.rows,c.cols)==(9,12)


def test_preview_draws_every_cell_and_highlights_a_model_pick():
    from ms.web_monitor import draw_board
    from unittest.mock import MagicMock
    cv=MagicMock()
    cv.winfo_width.return_value=300
    cv.winfo_height.return_value=200
    draw_board(cv,["1#","#F"],(1,2),[],set(),"orange")
    assert cv.create_rectangle.call_count==5 and cv.create_line.call_count==2


@pytest.fixture
def chrome_page():
    pw = pytest.importorskip("playwright.sync_api")
    with pw.sync_playwright() as api:
        try:
            browser = api.chromium.launch(channel="chrome")
        except pw.Error as exc:
            if (
                "not found" not in str(exc).lower()
                and "doesn't exist" not in str(exc).lower()
            ):
                raise
            pytest.skip("Chrome is not installed")
        try:
            page = browser.new_page()
            page.route(
                "http://grid.test/**",
                lambda route: route.fulfill(
                    content_type="text/html", body="<body></body>"
                ),
            )
            page.goto("http://grid.test/")
            yield page
        finally:
            browser.close()


def test_semantic_grid_capture_in_iframe_and_stale_or_obscured_click(chrome_page):
    from ms.web_browser import discover

    page = chrome_page
    page.set_content('<iframe style="width:310px;height:160px"></iframe>')
    child = page.frames[1]
    child.set_content(
        '<div role="grid" aria-rowcount="2" aria-colcount="3" style="display:grid;grid-template-columns:repeat(3,32px)">'
        + '<button role="gridcell" aria-label="covered" data-mine="1" style="height:30px" '
        'onclick="this.textContent=1" oncontextmenu="event.preventDefault();this.textContent=\'flagged\'"></button>'
        * 6
        + "</div>"
    )
    port = discover(lambda: [page], 0, 2)
    before = port.read()
    assert before.board.covered().all()
    port.click(before, (0, 0), "left")
    after = port.read()
    assert after.board.cells[0, 0] == 1
    with pytest.raises(Stuck):
        port.click(before, (0, 1), "left")
    port.click(after, (1, 1), "right")
    assert port.read().board.mines_left == 1
    child.locator("body").evaluate(
        """e => e.insertAdjacentHTML('beforeend','<div style="position:fixed;inset:0;background:white"></div>')"""
    )
    with pytest.raises(Unreadable):
        port.read()


def test_discovery_rejects_duplicate_boards_and_unknown_state(chrome_page):
    from ms.web_browser import discover

    page = chrome_page
    # Every real cell is a visible DOM node. A hidden mine attribute alone carries no clue.
    grid = (
        '<div role="grid" style="display:grid;grid-template-columns:repeat(2,25px)">'
        + (
            '<div role="gridcell" style="height:24px" data-mine="true" aria-label="unrecognised"></div>'
            * 4
        )
        + "</div>"
    )
    page.set_content(grid + grid)
    with pytest.raises(Stuck):
        discover(lambda: [page], 0, 1)
    page.locator('[role="grid"]').last.evaluate("e=>e.remove()")
    with pytest.raises(Unreadable):
        discover(lambda: [page], 0, 1).read()


def test_online_visible_tiles_counter_and_new_game_permalink(chrome_page):
    from ms.web_browser import discover

    page = chrome_page
    page.evaluate("history.replaceState(null,'','/game/111')")
    page.set_content("""
      <style>#CellsBlock{display:grid;grid-template-columns:repeat(2,28px)}
      .cell, i, button {width:28px;height:28px;display:inline-block}</style>
      <div id="game"><button id="top_area_face" class="hd_top-area-face-unpressed"></button>
      <i id="top_area_mines_100" class="hd_top-area-num0"></i><i id="top_area_mines_10" class="hd_top-area-num0"></i>
      <i id="top_area_mines_1" class="hd_top-area-num0"></i><div id="CellsBlock">
      <div id="cell_0_0" class="cell hd_closed hd_flag"></div><div id="cell_1_0" class="cell hd_opened hd_type2"></div>
      <div id="cell_0_1" class="cell hd_closed" data-mine="yes"></div><div id="cell_1_1" class="cell hd_opened hd_type1"></div>
      </div></div>""")
    port = discover(lambda: [page], 0, None)
    f = port.read()
    assert f.board.mines_left == 0 and f.board.cells.tolist() == [[-2, 2], [-1, 1]]
    page.locator("#top_area_face").evaluate("""e=>{
        e.className='hd_top-area-face-lose';
        e.onclick=()=>{
            history.pushState(null,'','/game/112');
            const g=document.querySelector('#game'), next=g.cloneNode(true); g.replaceWith(next);
            next.querySelectorAll('.cell').forEach(x=>x.className='cell hd_closed');
            next.querySelector('button').className='hd_top-area-face-unpressed';
            next.querySelector('#top_area_mines_1').className='hd_top-area-num1';
        }}""")
    assert port.read().status == "lost"
    assert port.restart().board.covered().all()
    page.evaluate("history.pushState(null,'','/game/113')")
    with pytest.raises(Stuck):
        port.read()
