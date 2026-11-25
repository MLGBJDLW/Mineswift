"""Measure perception against rendered boards and print numbers that can be quoted.

Run: python experiments/perception_probe.py

Every number here is a measurement of the current code on the synthetic world. They go
into the README verbatim; if a change moves them, the change has to justify itself.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ms.board import COVERED, FLAG, UNREADABLE, Unreadable   # noqa: E402
from ms.perception import Perception, Rect, fit_grid         # noqa: E402
from tests._synthetic import (Theme, make_scene, random_theme, render,   # noqa: E402
                              sweep)

THEMES = {
    "block light": dict(face="block", cell=30),
    "block small": dict(face="block", cell=20),
    "wide light": dict(face="wide", cell=30),
    "seg7 light": dict(face="seg7", cell=30),
}


def sloppy_rect(rng, scene, slack: float = 0.15) -> Rect:
    p = scene.pitch
    ox = float(rng.uniform(0.10, slack)) * p
    oy = float(rng.uniform(0.10, slack)) * p
    return Rect(
        int(round(scene.x0 - ox)),
        int(round(scene.y0 - oy)),
        int(round(30 * p + ox + float(rng.uniform(0.05, slack)) * p)),
        int(round(16 * p + oy + float(rng.uniform(0.05, slack)) * p)),
    )


def section(title: str) -> None:
    print(f"\n== {title} " + "=" * max(0, 58 - len(title)))


def geometry() -> None:
    section("geometry: worst error in px")
    print(f"  {'theme':<14s} {'tight':>10s} {'sloppy':>10s} {'tight aa':>10s}   n")
    rows = (("block", 20), ("block", 24), ("block", 30), ("wide", 30))
    for face, cell in rows:
        rng = np.random.default_rng(1)
        tight, loose, aa = [], [], []
        for _ in range(8):
            theme = random_theme(rng, cell=cell, face=face)
            if not (theme.separator or theme.bevel):
                continue
            scene = make_scene(rng, cell=cell, theme=theme)
            for rect, store in ((scene.rect, tight), (sloppy_rect(rng, scene), loose)):
                try:
                    cal = fit_grid(scene.fresh, rect, 16, 30)
                except Unreadable:
                    store.append(float("inf"))
                    continue
                store.append(max(abs(cal.x0 - scene.x0), abs(cal.y0 - scene.y0),
                                 abs(cal.pitch_x - scene.pitch), abs(cal.pitch_y - scene.pitch)))
            try:
                cal = fit_grid(scene.fresh, scene.rect, 16, 30)
                aa.append(max(abs(cal.pitch_x - scene.pitch), abs(cal.pitch_y - scene.pitch)))
            except Unreadable:
                aa.append(float("inf"))

        def worst(v):
            return "n/a" if not v or not np.isfinite(max(v)) else f"{max(v):.2f} px"

        print(f"  {face + ' ' + str(cell):<14s} {worst(tight):>10s} {worst(loose):>10s} "
              f"{worst(aa):>10s}   {len(tight)}")


def bevel_tight() -> None:
    section("geometry: bevel+separator themes, rect hugging the grid")
    print("  (this is the case that used to land one cell - 27..29 px - off)")
    for name, fn in (("pixel sweep", sweep), ("antialiased", render)):
        worst, n, fails = 0.0, 0, 0
        for seed in range(60):
            theme = random_theme(np.random.default_rng(seed), cell=30)
            if not (theme.bevel and theme.separator):
                continue
            n += 1
            fresh = np.full((16, 30), COVERED, dtype=np.int8)
            frame, (mx, my) = fn(fresh, theme)
            try:
                cal = fit_grid(frame, Rect(mx, my, 900, 480), 16, 30)
            except Unreadable:
                fails += 1
                continue
            err = max(abs(cal.x0 - mx), abs(cal.y0 - my))
            worst = max(worst, err)
            fails += err > 0.5
        print(f"  {name:<14s} worst {worst:.2f} px   over 0.5 px: {fails}/{n}")


def reading() -> None:
    section("reading: per-theme detection and digit agreement")
    print(f"  {'theme':<14s} {'covered':>11s} {'flags':>7s} {'digits ok':>10s} "
          f"{'wrong':>6s} {'unreadable':>11s}")
    for label, kw in THEMES.items():
        rng = np.random.default_rng(4)
        cov_ok = cov_n = fl_ok = fl_n = acc = wrong = unread = 0
        for _ in range(4):
            theme = random_theme(rng, cell=kw["cell"], face=kw["face"])
            if not (theme.separator or theme.bevel):
                continue
            scene = make_scene(rng, cell=kw["cell"], face=kw["face"], theme=theme)
            per = Perception(sloppy_rect(rng, scene), rows=16, cols=30)
            try:
                per.calibrate(scene.fresh)
                board = per.read(scene.frame, scene.mines_left, strict=False)
            except Unreadable:
                continue
            tc, tf = scene.cells == COVERED, scene.cells == FLAG
            rc, rf = board.cells == COVERED, board.cells == FLAG
            cov_ok += int(np.sum(rc == tc)); cov_n += tc.size
            fl_ok += int(np.sum(rf == tf)); fl_n += tf.size
            rev = scene.cells >= 0
            acc += int(np.sum((board.cells >= 0) & rev))
            wrong += int(np.sum((board.cells >= 0) & rev & (board.cells != scene.cells)))
            unread += int(np.sum((board.cells == UNREADABLE) & rev))
        print(f"  {label:<14s} {cov_ok:>6d}/{cov_n:<4d} {fl_ok:>3d}/{fl_n:<3d} "
              f"{acc:>10d} {wrong:>6d} {unread:>11d}")


def per_digit() -> None:
    section("reading: recall per digit (a missing glyph shows up here, not in the total)")
    seen = {}
    read = {}
    wrong = {}
    for seed in range(8):
        rng = np.random.default_rng(seed)
        scene = make_scene(rng, cell=30)
        per = Perception(scene.rect, rows=16, cols=30)
        try:
            per.calibrate(scene.fresh)
        except Unreadable:
            continue
        board = per.read(scene.frame, scene.mines_left, strict=False)
        rev = scene.cells >= 0
        for d in range(1, 9):
            t = rev & (scene.cells == d)
            seen[d] = seen.get(d, 0) + int(t.sum())
            read[d] = read.get(d, 0) + int((t & (board.cells == d)).sum())
            wrong[d] = wrong.get(d, 0) + int((t & (board.cells >= 0) & (board.cells != d)).sum())
    tot_s = sum(seen.values()); tot_r = sum(read.values())
    print(f"  {'digit':<6s} {''.join(f'{d:>8d}' for d in range(1, 9))}")
    print(f"  {'seen':<6s} {''.join(f'{seen.get(d, 0):>8d}' for d in range(1, 9))}")
    print(f"  {'read':<6s} {''.join(f'{read.get(d, 0):>8d}' for d in range(1, 9))}")
    print(f"  {'wrong':<6s} {''.join(f'{wrong.get(d, 0):>8d}' for d in range(1, 9))}")
    print(f"  overall recall {tot_r}/{tot_s} = {tot_r / max(1, tot_s):.1%}, "
          f"errors {sum(wrong.values())}")


def rejection() -> None:
    section("rejection: inputs that must raise Unreadable")
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 256, size=(500, 900, 3), dtype=np.uint8)
    flat = np.full((500, 900, 3), 128, dtype=np.uint8)
    lattice_free = random_theme(np.random.default_rng(9), cell=30)
    fresh = np.full((16, 30), COVERED, dtype=np.int8)
    sep_frame, (sx, sy) = sweep(fresh, lattice_free)
    scene = make_scene(np.random.default_rng(5), cell=30)
    pale = (scene.fresh.astype(np.float32) * 0.04 + 128 * 0.96).astype(np.uint8)

    cases = [
        ("random noise", lambda: fit_grid(noise, Rect(20, 20, 800, 440), 16, 30)),
        ("flat colour", lambda: fit_grid(flat, Rect(20, 20, 800, 440), 16, 30)),
        ("no lattice on screen", lambda: fit_grid(sep_frame, Rect(sx, sy, 900, 480), 16, 30)),
        ("wrong row count", lambda: fit_grid(scene.fresh, scene.rect, 9, 30)),
        ("washed out (4% contrast)",
         lambda: Perception(scene.rect, rows=16, cols=30).calibrate(pale)),
    ]
    for name, fn in cases:
        try:
            fn()
            print(f"  {name:<26s} NOT rejected")
        except Unreadable as exc:
            print(f"  {name:<26s} rejected: {str(exc)[:52]}")


def frame_agreement() -> None:
    section("frame agreement: what read_stable refuses")
    theme = Theme(
        cell=30, covered=(176, 176, 176), covered_light=(200, 200, 200),
        covered_dark=(150, 150, 150), bg=(235, 235, 235), ink=(40, 40, 40),
        grid=(110, 110, 110), flag=(200, 40, 40), bevel=False, separator=True, margin=16,
    )
    scene = make_scene(np.random.default_rng(9), theme=theme)
    per = Perception(scene.rect, rows=16, cols=30)
    per.calibrate(scene.fresh)
    for label, frames in (
        ("identical frames", [scene.frame, scene.frame.copy()]),
    ):
        try:
            per.read_stable(frames, scene.mines_left)
            print(f"  {label:<26s} accepted (expected)")
        except Unreadable as exc:
            print(f"  {label:<26s} rejected: {str(exc)[:52]}")

    moved = scene.frame.copy()
    rc = tuple(int(v) for v in np.argwhere(scene.cells >= 0)[0])
    top = int(scene.y0) + rc[0] * 30 + 10
    left = int(scene.x0) + rc[1] * 30 + 10
    moved[top:top + 4, left:left + 4] = 0
    try:
        per.read_stable([scene.frame, moved], scene.mines_left)
        print(f"  {'one cell changed':<26s} accepted (BAD)")
    except Unreadable as exc:
        print(f"  {'one cell changed':<26s} rejected: {str(exc)[:52]}")


def timing() -> None:
    section("timing")
    import time as _t
    rng = np.random.default_rng(2)
    # a theme with visible separators, so the fit has something to find
    while True:
        theme = random_theme(rng, cell=30)
        if theme.separator or theme.bevel:
            break
    scene = make_scene(rng, cell=30, theme=theme)
    per = Perception(scene.rect, rows=16, cols=30)

    t0 = _t.perf_counter()
    for _ in range(10):
        fit_grid(scene.fresh, scene.rect, 16, 30)
    fit_ms = (_t.perf_counter() - t0) / 10 * 1000

    t0 = _t.perf_counter()
    per.calibrate(scene.fresh)
    calib_ms = (_t.perf_counter() - t0) * 1000

    t0 = _t.perf_counter()
    for _ in range(20):
        per.read(scene.frame, scene.mines_left, strict=False)
    read_ms = (_t.perf_counter() - t0) / 20 * 1000

    print(f"  fit_grid           {fit_ms:6.1f} ms")
    print(f"  calibrate          {calib_ms:6.1f} ms  (fit + palette)")
    print(f"  read (480 cells)   {read_ms:6.1f} ms  per frame")


def main() -> int:
    geometry()
    bevel_tight()
    reading()
    per_digit()
    rejection()
    frame_agreement()
    timing()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
