"""Play a visible web or desktop Minesweeper board with a trained policy."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, nullcontext
from datetime import datetime
import hashlib
import json
from pathlib import Path
import secrets
import sys
import time
from threading import Event

import numpy as np

from .board import Board, COVERED, FLAG, Stuck, Unreadable
from .chording import chord_candidates
from .solver import NEIGHBOURS, ORTHOGONAL, Solver, _nbrs, raw_constraints
from .web_chrome import BoardChanged, ChromeGame, ReadbackError, RestartRound, StopRequested, settled
from .web_vision import Sprites

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "models" / "belief_rl.pt"
VARIANT_WEIGHTS = ROOT / "models" / "rl_guess.pt"


def load_policy(seed, rules="classic", *, review=True):
    import torch
    from .belief_rl import BeliefActor
    from .belief_search import RolloutActor
    from .verified_search import VerifiedActor

    torch.set_num_threads(1)
    if rules == "orthogonal":
        from .rl import BoardPolicyNet, RLGuessPolicy
        net = BoardPolicyNet()
        net.load_state_dict(torch.load(VARIANT_WEIGHTS, weights_only=True, map_location="cpu")["model"])
        policy = RLGuessPolicy(seed, net.eval())
        policy.solver = Solver(seed, max_nodes=None, offsets=ORTHOGONAL)
        return policy
    net = BeliefActor()
    net.load_state_dict(torch.load(WEIGHTS, weights_only=True, map_location="cpu")["model"])
    return (VerifiedActor if review else RolloutActor)(net.eval(), seed)


def visible_rows(board):
    symbols = {COVERED: "#", FLAG: "F"}
    return ["".join(symbols.get(int(v), str(v)) for v in row) for row in board.cells]


def verify_change(before, after, cell, action, targets=(), *, during_update=False):
    if after.status != "playing":
        # The score panel can shift the entire grid on the winning click.
        if before.layout != after.layout and before.layout.geometry()[:2] != after.layout.geometry()[:2]:
            raise Stuck("终局的棋盘尺寸发生变化。")
        if after.status == "lost" and action in ("flag", "unflag", "open", "chord"):
            raise Stuck("确定步后读到失败表情，请检查本局日志。")
        return
    if before.layout != after.layout:
        raise Stuck("操作过程中棋盘位置或尺寸发生变化。")
    a, b = before.board, after.board
    changed = a.cells != b.cells
    reveals = (a.cells == COVERED) & (b.cells >= 0)
    if action in ("flag", "unflag"):
        target, delta = (FLAG, -1) if action == "flag" else (COVERED, 1)
        other_changes = changed & ~reveals
        other_changes[cell] = False
        possible = not other_changes.any() and b.cells[cell] in (a.cells[cell], target)
        ok = possible and b.cells[cell] == target and b.mines_left == a.mines_left + delta
        pending = possible and b.mines_left in (a.mines_left, a.mines_left + delta)
    else:
        opens = targets if action == "chord" else (cell,)
        pending = b.mines_left == a.mines_left and np.all(~changed | reveals)
        ok = pending and bool(opens) and all(reveals[p] for p in opens)
    if not ok:
        if during_update and pending:
            raise Unreadable("正在等待本次落子的目标格和计数器完成更新")
        raise Stuck("点击后的格子或雷数与预期不符。")


def sync_before_input(port, frame, read_after):
    current = read_after(port)
    if (current.status == "playing" and current.layout == frame.layout
            and current.board.mines_left == frame.board.mines_left):
        changed = frame.board.cells != current.board.cells
        if changed.any() and np.all((frame.board.cells[changed] == COVERED) & (current.board.cells[changed] >= 0)):
            return current
    if current.status == "won" and current.layout.geometry() == frame.layout.geometry():
        return current
    raise Stuck("落子前局面已改变，无法确认是同一盘的连续展开。")


def click_and_observe(port, frame, cell, button, read_after, validate, emit, *, retry=False,
                      unchanged_timeout=None):
    started = time.perf_counter()
    port.click(frame, cell, button)
    sent = time.perf_counter()
    row, col = cell[0] + 1, cell[1] + 1
    emit("input_sent", row=row, col=col, button=button, attempt=2 if retry else 1,
         dispatch_ms=round((sent - started)*1000, 1))
    try:
        options = {} if unchanged_timeout is None else {"unchanged_timeout": unchanged_timeout}
        return read_after(port, before=frame, allow_unchanged=True, validate=validate, **options)
    except ReadbackError as exc:
        changed = port.target_changed(cell) if hasattr(port, "target_changed") else None
        emit("input_unconfirmed", row=row, col=col, button=button, target_pixels_changed=changed,
             feedback_ms=round((time.perf_counter()-sent)*1000, 1))
        key = {"left": "左键", "right": "右键", "chord": "双键"}[button]
        detail = {True: "目标格画面已变", False: "目标格画面未变", None: "尚未确认画面变化"}[changed]
        raise ReadbackError(f"第 {row} 行第 {col} 列{key}已发出，{detail}。{exc}") from exc


def best_chord(board, physical, safe):
    safe = set(safe)
    if len(safe) < 2:
        return None
    # The logical board treats unverified human flags as covered. Requiring every
    # target to be physically covered excludes anchors that touch such a flag.
    return max(((anchor, opens) for anchor, opens in chord_candidates(board)
                if board.cells[anchor] > 0 and len(opens) >= 2
                and all(p in safe and physical.cells[p] == COVERED for p in opens)),
               key=lambda item: len(item[1]), default=None)


def economical_chord(board, physical, safe, flag_cost=1.0, chord_cost=1.0):
    """Compare confirmed input costs in units of one direct reveal."""
    safe = set(safe)
    best, score = None, (0, 0)
    for anchor, opens in chord_candidates(board):
        if board.cells[anchor] <= 0 or len(opens) < 2 or not safe.intersection(opens):
            continue
        if any(physical.cells[p] != COVERED for p in opens):
            continue
        missing = [p for p in _nbrs(board.rows, board.cols, *anchor)
                   if board.cells[p] == FLAG and physical.cells[p] == COVERED]
        gain = len(opens) - chord_cost - len(missing) * flag_cost
        if gain > 0 and (gain, -len(missing)) > score:
            score = (gain, -len(missing))
            best = ("flag", missing[0], ()) if missing else ("chord", anchor, opens)
    return best


def explain(board, cell, action, targets, details):
    if action == "opening":
        return "棋盘还没有数字，均匀随机选择首点。"
    if action == "guess":
        if details.get("verified_worlds"):
            return (f"没有确定步。复核了 {details['verified_worlds']} 种相容雷图，"
                    f"选这格后，后续走最优步可以赢其中 {details['optimal_wins']} 盘。")
        worlds = details.get("rollout_worlds", 0)
        if worlds:
            return f"没有确定步。选这格后，模型在 {worlds} 种相容雷局中试走赢了 {details['rollout_wins']} 局。"
        return "没有确定步，训练模型根据整盘数字和各格雷率选择了这格。"
    if action == "unflag":
        return "旧旗未经证实，先取下，再读盘决策。"
    if action == "chord":
        return f"数字 {board.cells[cell]} 周围的雷已全部证实并插旗，其余 {len(targets)} 格安全。"
    for p in _nbrs(board.rows, board.cols, *cell):
        clue = int(board.cells[p])
        if clue < 0:
            continue
        adj = [board.cells[q] for q in _nbrs(board.rows, board.cols, *p)]
        flags, covered = adj.count(FLAG), adj.count(COVERED)
        prefix = f"第 {p[0]+1} 行第 {p[1]+1} 列是 {clue}，"
        if action == "open" and flags == clue:
            return prefix + f"周围已有 {flags} 面确定旗，其余格安全。"
        if action == "flag" and clue - flags == covered:
            return prefix + f"还缺 {covered} 颗雷，正好剩 {covered} 格。"
    return "结合整盘数字与剩余雷数，所有相容布局中这格" + ("都是雷。" if action == "flag" else "都安全。")


def play(port, policy, frame, seed, emit, *, chords=True, blind=False, hybrid=False,
         adapt_to_latency=False, read_after=settled, rules="classic"):
    rng = np.random.default_rng(seed)
    counts = Counter()
    durations = {}
    slowest_confirmed_input = 0.0
    adapt_to_latency = adapt_to_latency and rules == "classic" and not blind
    open_samples = 0
    flag_cost = chord_cost = 1.0
    if frame.status == "playing":
        trusted = np.zeros(frame.board.cells.shape, dtype=bool)
        limit = frame.board.cells.size * 4 + 2
    else:
        limit = 0
    planning_started = time.perf_counter()
    for _ in range(limit):
        if frame.status != "playing":
            break
        physical = frame.board
        targets = ()
        details = {}
        cells = physical.cells.copy()
        # A flag inherited from the human is an annotation until this solver proves it.
        unproved = physical.flags() & ~trusted
        remembered = trusted & physical.covered()
        if np.any(trusted & (physical.cells >= 0)):
            raise Stuck("读盘发现已证实的雷被翻开。")
        cells[unproved] = COVERED
        cells[remembered] = FLAG
        board = Board(physical.rows, physical.cols, cells,
                      physical.mines_left + int(unproved.sum()) - int(remembered.sum()))
        if not 0 <= board.mines_left <= int(board.covered().sum()):
            raise Stuck("屏幕计数器与记忆中的雷数不一致。")
        raw_constraints(board, strict=True, offsets=ORTHOGONAL if rules=="orthogonal" else NEIGHBOURS)
        if np.all(cells == COVERED) and physical.covered().any():
            cell = divmod(int(rng.choice(np.flatnonzero(physical.covered()))), board.cols)
            source, action = "opening", "opening"
            reason = "uniform random first click"
        else:
            move = policy.act(board)
            if move.kind not in ("flag", "open", "guess") or not move.cells:
                raise Stuck("求解器已无落子，但网页仍显示进行中。")
            cell = move.cells[0]
            if not board.covered()[cell]:
                raise Stuck("策略选择了不可操作的格子。")
            source = "model" if move.kind == "guess" else "solver"
            action = move.kind
            reason = move.reason
            if action == "flag" and (blind or hybrid):
                for p in move.cells:
                    trusted[p] = True
                emit("remembered", cells=[(r+1,c+1) for r,c in move.cells])
                continue
            if action == "guess":
                details = getattr(policy, "last_guess", {})
            if action == "flag" and unproved[cell]:
                trusted[cell] = True
                emit("confirmed_flag", row=cell[0]+1, col=cell[1]+1)
                continue
            if unproved[cell]:
                action = "unflag"
            if rules == "classic" and chords and not blind and move.kind == "open":
                if hybrid:
                    planned = economical_chord(board, physical, move.cells, flag_cost, chord_cost)
                    if planned:
                        action, cell, targets = planned
                else:
                    planned = best_chord(board, physical, move.cells)
                    if planned:
                        cell, targets = planned
                        action = "chord"
        emit("decision", source=source, action=action, row=cell[0]+1, col=cell[1]+1,
             planning_ms=round((time.perf_counter()-planning_started)*1000, 1),
             reason=reason, mines_left=physical.mines_left, board=visible_rows(physical),
             targets=[(r+1,c+1) for r,c in targets], clicks=dict(counts),
             memory=[(int(r)+1,int(c)+1) for r,c in zip(*np.nonzero(remembered))],
             explanation=(explain(board, cell, action, targets, details) if rules=="classic" else
                          {"guess":"四邻域没有确定步，由整盘模型选格。","opening":"均匀随机选择首点。",
                           "unflag":"移除未经证实的旗。","flag":"四邻域约束证明此格是雷。",
                           "open":"四邻域约束证明此格安全。"}[action]), **details)
        button = {"chord": "chord", "flag": "right", "unflag": "right"}.get(action, "left")
        started = time.perf_counter()
        def validate_after(candidate):
            verify_change(frame, candidate, cell, action, targets, during_update=True)

        try:
            probe = None
            if action == "chord" and not counts["chords"] and slowest_confirmed_input:
                probe = max(.4, min(3., 4 * slowest_confirmed_input))
            after = click_and_observe(port, frame, cell, button, read_after, validate_after, emit,
                                      unchanged_timeout=probe)
        except BoardChanged:
            frame = sync_before_input(port, frame, read_after)
            emit("resynced", status=frame.status)
            planning_started = time.perf_counter()
            continue
        unchanged = after.matches(frame)
        if action == "chord" and unchanged:
            chords = False
            emit("chord_unavailable", row=cell[0]+1, col=cell[1]+1,
                 input_to_read_ms=round((time.perf_counter()-started)*1000, 1))
            planning_started = time.perf_counter()
            continue
        if unchanged:
            if action == "flag":
                # A clean timeout on a proved mine: leave it covered physically.
                # Suppress chords so no gesture can rely on a missing physical flag.
                trusted[cell] = True
                blind, chords = True, False
                emit("flag_unavailable", row=cell[0]+1, col=cell[1]+1)
                planning_started = time.perf_counter()
                continue
            if button != "left":
                raise Stuck("右键发出后整盘持续无变化，请检查游戏的标旗设置。")
            # Only repeat a left click after a clean unchanged window. Each port
            # rechecks the target window and full board immediately before input.
            emit("input_retry", row=cell[0]+1, col=cell[1]+1)
            try:
                duration = port.longer_left_press() if hasattr(port, "longer_left_press") else nullcontext()
                with duration:
                    after = click_and_observe(port, frame, cell, button, read_after, validate_after, emit, retry=True)
                    if after.matches(frame):
                        raise Stuck(f"第 {cell[0]+1} 行第 {cell[1]+1} 列再次左击后整盘仍无变化。")
                    verify_change(frame, after, cell, action, targets)
            except BoardChanged:
                frame = sync_before_input(port, frame, read_after)
                emit("resynced", status=frame.status)
                planning_started = time.perf_counter()
                continue
        verify_change(frame, after, cell, action, targets)
        elapsed = time.perf_counter() - started
        slowest_confirmed_input = max(slowest_confirmed_input, elapsed)
        if adapt_to_latency and chords and not unchanged and action in ("flag", "open", "chord"):
            durations[action] = .7 * durations.get(action, elapsed) + .3 * elapsed
            open_samples += int(action == "open")
            if open_samples >= 2 and "flag" in durations:
                reveal = durations["open"]
                # Only change planning on boards with expensive reveals and
                # substantially cheaper flags. Once virtual flags exist, keep
                # using the planner that checks all required physical flags.
                if reveal >= .15 and durations["flag"] < reveal / 2:
                    hybrid = True
                    flag_cost = durations["flag"] / reveal
                    chord_cost = max(1., durations.get("chord", reveal) / reveal)
                else:
                    flag_cost = chord_cost = 1.0
        counts["unflag" if action == "unflag" else source] += 1
        if action == "chord":
            counts["chords"] += 1
            counts["chord_targets"] += len(targets)
        if action == "flag":
            trusted[cell] = True
        frame = after
        emit("observed", status=frame.status, clicks=dict(counts),
             input_to_read_ms=round(elapsed*1000, 1),
             board=visible_rows(frame.board) if frame.board is not None else None)
        planning_started = time.perf_counter()
    if frame.status == "playing":
        raise Stuck("已达到本盘最大操作数。")
    emit("finished", status=frame.status, clicks=dict(counts))
    return frame.status, counts


class Controls:
    def __init__(self):
        self.stop, self.again, self.auto = Event(), Event(), Event()
        self.blind = Event()
        self.panel_hwnd = 0


def run_rounds(port, frame, seed, emit, *, controls=None, games=None, chords=True, blind=False, hybrid=False, rules="classic"):
    completed = 0
    number = 0
    while True:
        number += 1
        round_seed = seed + number - 1
        rows = visible_rows(frame.board) if frame.board is not None else []
        bounds = port.screen_bounds(frame.layout) if hasattr(port,"screen_bounds") else frame.layout.bounds()
        emit("layout", bounds=bounds)
        round_blind = controls.blind.is_set() if controls else blind
        emit("attached", round=number, seed=round_seed, status=frame.status, board=rows, blind=round_blind,
             mines_left=frame.board.mines_left if rows else None)
        try:
            if frame.status == "playing":
                policy = load_policy(round_seed) if rules=="classic" else load_policy(round_seed,rules)
                weights = WEIGHTS if rules=="classic" else VARIANT_WEIGHTS
                emit("policy_loaded", policy=policy.name, weights=weights.name,
                     sha256=hashlib.sha256(weights.read_bytes()).hexdigest())
                use_hybrid = (hybrid and rules == "classic" and not round_blind and chords
                              and frame.board.cells.shape == (9, 9)
                              and frame.board.mines_left + int(frame.board.flags().sum()) == 10)
                status, counts = play(port, policy, frame, round_seed, emit, chords=chords, blind=round_blind,
                                      hybrid=use_hybrid, adapt_to_latency=hybrid and rules=="classic", rules=rules)
                completed += 1
                if games is not None and completed >= games:
                    return
            else:
                status, counts = frame.status, {}
            if controls is None:
                return
            emit("waiting", status=status, clicks=dict(counts))
            start = time.monotonic()
            while True:
                port.guard()
                if (controls.auto.is_set() or games is not None) and time.monotonic() - start >= 1.25:
                    break
                time.sleep(.05)
        except RestartRound:
            emit("restart_requested", round=number)
        frame = port.restart()
        emit("restarted", next_seed=round_seed+1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--url", help="用独立 Chrome 打开此网址，自动识别棋盘")
    source.add_argument("--cdp", help="连接本机已开启调试的 Chrome，识别可见标签")
    source.add_argument("--desktop", action="store_true", help="用 F8 框选桌面程序或普通浏览器里的棋盘")
    source.add_argument("--window", help="在指定完整标题的桌面窗口中自动找盘，配合 --shape 行 列")
    parser.add_argument("--region", type=int, nargs=4, metavar=("X","Y","W","H"), help="桌面棋盘的物理像素范围")
    parser.add_argument("--shape", type=int, nargs=2, metavar=("ROWS","COLS"), help="桌面棋盘行数和列数")
    parser.add_argument("--covered", type=int, nargs=2, metavar=("ROW","COL"), help="框内的已知未开格，行列从 1 开始")
    parser.add_argument("--tiles", type=Path, help="桌面皮肤 PNG 图块目录")
    parser.add_argument("--classic-colors", action="store_true", help="辅助识别经典灰盘上的红旗和触雷图标")
    parser.add_argument("--sample-tile", action="append", nargs=3, metavar=("LABEL","ROW","COL"),
                        help="从框内指定格保存本皮肤图块，可重复使用；需要 --inspect --tiles")
    parser.add_argument("--mines", type=int, help="桌面或无计数器的语义网格的总雷数")
    parser.add_argument("--rules", choices=("classic","orthogonal"), default="classic", help="八邻域或上下左右四邻域")
    parser.add_argument("--restart-point", type=int, nargs=2, metavar=("X","Y"), help="桌面游戏重开按钮的物理坐标")
    parser.add_argument("--inspect", action="store_true", help="识别当前局面并退出")
    parser.add_argument("--wait", type=float, default=15, metavar="SECONDS", help="等待已打开的扫雷标签")
    parser.add_argument("--seed", type=int, default=None, help="随机首点种子")
    parser.add_argument("--watch", action="store_true", help="打开实时决策窗，终局后等待 R 或自动续局")
    parser.add_argument("--games", type=int, metavar="N", help="连续完成 N 局后退出")
    parser.add_argument("--no-chord", action="store_true", help="仅使用逐格点击")
    parser.add_argument("--blind", action="store_true", help="盲扫：把确定的雷记在内存中，省去插旗")
    args = parser.parse_args()
    desktop = args.desktop or args.window is not None or args.region is not None
    if args.window and not args.shape:
        parser.error("--window 自动找盘需要 --shape 行 列")
    if args.region and (args.url or args.cdp or args.window):
        parser.error("--region 只能用于桌面捕获")
    if (args.tiles or args.shape or args.covered or args.restart_point or args.classic_colors) and not desktop:
        parser.error("桌面参数需要 --desktop 或 --region")
    if args.sample_tile and not (desktop and args.inspect and args.tiles):
        parser.error("--sample-tile 需要桌面模式、--inspect 和 --tiles 目录")
    if args.mines is not None and args.mines < 1:
        parser.error("--mines 需要正整数")
    if desktop and args.mines is None:
        parser.error("桌面模式需要 --mines 指定游戏的总雷数")
    if args.rules!="classic" and not (desktop or args.url or args.cdp):
        parser.error("变种规则需要显式选择捕获入口")
    if args.games is not None and args.games < 1:
        parser.error("--games 需要正整数")
    if args.inspect and (args.games or args.watch):
        parser.error("--inspect 单独使用")
    controls = Controls() if args.watch or args.games else None
    if controls and args.blind:
        controls.blind.set()
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    directory = ROOT / "reports" / "web"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".jsonl")
    print(f"正在寻找扫雷棋盘。Esc 停止。\n本次记录：{path}", flush=True)
    def work(display):
        with path.open("x", encoding="utf-8") as log:
            def emit(event, **values):
                print(json.dumps(dict(event=event, **values), ensure_ascii=False), file=log, flush=True)
                display(event, values)
                if event == "decision":
                    print(f"{values['action']}  第 {values['row']} 行，第 {values['col']} 列", flush=True)
                if event in ("finished", "waiting"):
                    print(f"{'胜利' if values['status']=='won' else '踩雷'}  {values['clicks']}", flush=True)
            try:
                with ExitStack() as stack:
                    if not args.inspect and sys.platform=="win32":
                        stack.enter_context(ChromeGame.one_player())
                    if args.url or args.cdp:
                        from .web_browser import browser_session
                        port = stack.enter_context(browser_session(args.url,args.cdp,max(0,args.wait),args.mines))
                    elif desktop:
                        from .desktop_grid import DesktopGame
                        port = DesktopGame(args)
                    else:
                        if sys.platform!="win32":
                            raise Stuck("普通 Chrome 接盘需要 Windows，另可使用 --url。")
                        port = ChromeGame(Sprites.load())
                        port.attach(max(0,args.wait))
                    emit("capture", source=getattr(port,"name","chrome-canvas"), rules=args.rules)
                    if args.sample_tile:
                        print(f"已向 {args.tiles} 保存 {len(args.sample_tile)} 个图块。")
                        return 0
                    frame = settled(port)
                    if args.inspect:
                        rows = visible_rows(frame.board) if frame.board is not None else []
                        emit("inspected",status=frame.status,board=rows,mines_left=frame.board.mines_left if rows else None)
                        print("\n".join(rows) or frame.status)
                        return 0
                    port.controls = controls
                    run_rounds(port, frame, seed, emit, controls=controls, games=args.games,
                               chords=not args.no_chord, blind=args.blind, rules=args.rules)
                return 0
            except StopRequested as exc:
                emit("stopped", reason=str(exc))
                return 0
            except (Exception, KeyboardInterrupt) as exc:
                message = str(exc) or type(exc).__name__
                emit("stopped", reason=message)
                print(f"停止：{message}", file=sys.stderr)
                return 2
    if args.watch and sys.platform == "win32":
        from .web_monitor import watch
        return watch(work, controls, args.games is not None)
    return work(lambda *_: None)


if __name__ == "__main__":
    raise SystemExit(main())
