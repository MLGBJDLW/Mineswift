"""Deterministic Minesweeper solver.

Contract: Solver.decide(board) -> Move. Same Board + same seed -> same Move, always.

Decision order (and nothing else):
    1. constraint propagation proves a cell safe   -> open it   (risk 0)
    2. constraint propagation proves a cell a mine -> flag it   (risk 0)
    3. otherwise                                   -> guess the cell with the lowest exact
                                                      global mine probability

Exactness: every independent frontier component is enumerated and turned into a histogram
of "solutions with exactly m mines" plus per-cell mine counts. Components are convolved
against the remaining mine budget and the free (non-frontier) cells are folded in with a
binomial term. That yields a true global probability per cell - the only defensible
meaning of "best guess", because the mine counter links components together.

enumerate_group counts assignments of the cells the group's constraints mention. Cells
mentioned by no constraint are not part of that count and are reported as safe; the
probability engine handles them through the free-cell term. This keeps "number of
solutions" exact instead of multiplying counts by 2^k for unconstrained cells.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from math import comb
from typing import Callable, Literal

import numpy as np

from .board import COVERED, FLAG, QUESTION, Board

SAMPLE_SECONDS = 0.15  # wall-clock ceiling for the sampling fallback, per component

NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
ORTHOGONAL = ((-1,0), (0,-1), (0,1), (1,0))


@dataclass(frozen=True)
class Move:
    kind: Literal["open", "flag", "chord", "guess", "done"]
    cells: tuple[tuple[int, int], ...]
    p_mine: float | None  # None for policies that do not estimate mine risk
    reason: str


Cell = tuple[int, int]
GuessRanker = Callable[[Board, Cell, dict[Cell, int], int], Cell]


@dataclass(frozen=True)
class Constraint:
    cells: tuple[int, ...]  # cell ids (r * cols + c)
    mines: int


ComponentCache = dict[tuple[Constraint, ...],
                      tuple[list[int], list[dict[int, int]], dict[int, int], int, int]]


class BudgetExceeded(Exception):
    """The exact search tree grew past the latency budget; the caller must degrade."""


class Inconsistent(Exception):
    """No assignment satisfies the constraints: perception read something wrong."""


def _nbrs(rows: int, cols: int, r: int, c: int, offsets=NEIGHBOURS):
    for dr, dc in offsets:
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            yield rr, cc


# --------------------------------------------------------------------------- #
# constraints from a visible board
# --------------------------------------------------------------------------- #
def raw_constraints(board: Board, *, strict: bool = False, offsets=NEIGHBOURS):
    """One constraint per revealed number: (covered neighbours, number - flags)."""
    rows, cols = board.rows, board.cols
    cons: list[Constraint] = []
    frontier: set[int] = set()
    pins: dict[int, bool] = {}
    for r in range(rows):
        for c in range(cols):
            v = int(board.cells[r, c])
            if v < 0:
                continue
            unk, flags = [], 0
            for rr, cc in _nbrs(rows, cols, r, c, offsets):
                x = int(board.cells[rr, cc])
                if x == FLAG:
                    flags += 1
                elif x == COVERED or x == QUESTION:
                    unk.append(rr * cols + cc)
            need = v - flags
            if strict:
                if not 0 <= need <= len(unk):
                    raise Inconsistent(f"clue at {(r, c)} needs {need} mines among {len(unk)} cells")
                if need == 0 or need == len(unk):
                    pin = need != 0
                    for cell in unk:
                        if pins.setdefault(cell, pin) != pin:
                            raise Inconsistent("two clues force opposite values on a cell")
            if unk:
                cons.append(Constraint(tuple(unk), need))
                frontier.update(unk)
    all_cov = {
        r * cols + c
        for r in range(rows)
        for c in range(cols)
        if int(board.cells[r, c]) in (COVERED, QUESTION)
    }
    return cons, frontier, all_cov - frontier


def certain_moves(raw: list[Constraint]) -> tuple[set[int], set[int]]:
    """Cells a single constraint already decides. Sound, cheap, no search."""
    safe: set[int] = set()
    mines: set[int] = set()
    for c in raw:
        if c.mines == 0:
            safe.update(c.cells)
        elif c.mines == len(c.cells):
            mines.update(c.cells)
    return safe - mines, mines


def simplify(raw: list[Constraint]) -> list[Constraint]:
    """The two classic rules to a fixpoint, used only to shrink the search.

    Rule A (decided facts): k == 0 makes the cells safe, k == len(cells) makes them mines;
    those cells are removed from every constraint and the mine count corrected.
    Rule B (subset subtraction): if cells(A) subset cells(B) then (B - A, kB - kA).

    Decisions never rely on this, so a missed deduction costs speed, not correctness.
    Raises Inconsistent when the rules contradict each other.
    """
    cons = {frozenset(c.cells): c for c in raw if c.cells}
    for _ in range(64):
        safe: set[int] = set()
        mine: set[int] = set()
        for f, c in cons.items():
            if c.mines == 0:
                safe.update(f)
            elif c.mines == len(f):
                mine.update(f)
        if safe & mine:
            raise Inconsistent("a cell is forced both safe and mine")
        changed = False
        if safe or mine:
            reduced: dict[frozenset, Constraint] = {}
            for c in cons.values():
                rest = tuple(x for x in c.cells if x not in safe and x not in mine)
                k = c.mines - sum(1 for x in c.cells if x in mine)
                if rest:
                    reduced[frozenset(rest)] = Constraint(rest, k)
            cons, changed = reduced, True
        keys = sorted(cons, key=len)
        for a in keys:
            ca = cons.get(a)
            if ca is None:
                continue
            for b in keys:
                if a < b:
                    cb = cons.get(b)
                    if cb is None:
                        continue
                    rest = tuple(x for x in cb.cells if x not in a)
                    k = cb.mines - ca.mines
                    f = frozenset(rest)
                    if 0 <= k <= len(rest) and f not in cons:
                        cons[f] = Constraint(rest, k)
                        changed = True
        if not changed:
            break
        if len(cons) > 3000:  # adversarial blow-up guard
            break
    return [c for c in cons.values() if c.cells]


# --------------------------------------------------------------------------- #
# exact counting over one frontier component
# --------------------------------------------------------------------------- #
def _reduce(cur, value: list[int], undo: list[int]):
    """Decide whatever `cur` forces under the current `value` assignment.

    Returns the reduced constraint list, [] when nothing is left to decide, or None when
    the system became contradictory. Implicit decisions are appended to `undo`.
    """
    work = cur
    while True:
        out = []
        f0: list[int] = []
        f1: list[int] = []
        for cs, k in work:
            und = [i for i in cs if value[i] < 0]
            need = k - sum(1 for i in cs if value[i] == 1)
            if need < 0 or need > len(und):
                return None
            if und:
                out.append((tuple(und), need))
                if need == 0:
                    f0.extend(und)
                elif need == len(und):
                    f1.extend(und)
        if not out:
            return []
        if not f0 and not f1:
            return out
        if set(f0) & set(f1):
            return None
        for i in f0:
            if value[i] == 1:
                return None
            value[i] = 0
            undo.append(i)
        for i in f1:
            if value[i] == 0:
                return None
            value[i] = 1
            undo.append(i)
        work = out


def _rollback(value: list[int], undo: list[int], mark: int) -> None:
    while len(undo) > mark:
        value[undo.pop()] = -1


def enumerate_group(group: list[Constraint], max_nodes: int | None, *,
                    on_solution: Callable[[int], None] | None = None):
    """Exact counting on ONE component. Returns (cells, succ, hist, n_sol, nodes).

    Optional visitor receives each solution as a mine bitmask in sorted cell order.

    succ[j]  = number of solutions where cells[j] is a mine
    hist[m]  = number of solutions with exactly m mines
    n_sol    = total number of solutions (= sum(hist.values()))
    nodes    = search nodes used

    Counting is over the cells that constrain each other; a cell no constraint mentions is
    another component's business (or genuinely free) and reports succ 0, so the count stays
    exact instead of multiplying by 2^k.

    Fully decided constraints (k == 0 or k == |cells|) pin their cells outright. They are
    folded in first because their cells then need no branching:
      * if nothing undecided is left, a single solution is recorded;
      * otherwise the DFS runs and every recorded solution already includes those pins.

    Raises BudgetExceeded past max_nodes, Inconsistent when nothing satisfies the set.
    """
    merged: dict[tuple[int, ...], int] = {}
    for c in group:
        key = tuple(sorted(c.cells))
        if key in merged and merged[key] != c.mines:
            raise Inconsistent("two constraints disagree on the same cells")
        merged[key] = c.mines
    cells = sorted({x for key in merged for x in key})
    idx = {c: i for i, c in enumerate(cells)}
    n = len(cells)

    succ_by_k: list[dict[int, int]] = [dict() for _ in range(n)]  # cell -> {k: mine worlds}
    hist: dict[int, int] = {}
    nodes = 0
    value = [-1] * n  # -1 undecided, 0 safe, 1 mine
    undo: list[int] = []

    open_cons: list[tuple[tuple[int, ...], int]] = []
    for key, k in merged.items():
        local = tuple(idx[x] for x in key)
        if k == 0:
            for i in local:
                value[i] = 0
        elif k == len(local):
            for i in local:
                value[i] = 1
        else:
            open_cons.append((local, k))

    def record() -> None:
        if on_solution is not None:
            on_solution(sum(1 << j for j in range(n) if value[j] == 1))
        m = 0
        for j in range(n):
            if value[j] == 1:
                m += 1
        hist[m] = hist.get(m, 0) + 1
        for j in range(n):
            if value[j] == 1:
                bucket = succ_by_k[j]
                bucket[m] = bucket.get(m, 0) + 1

    def dfs(cur) -> None:
        nonlocal nodes
        nodes += 1
        if max_nodes is not None and nodes > max_nodes:
            raise BudgetExceeded
        mark = len(undo)
        cur = _reduce(cur, value, undo)
        if cur is None:
            _rollback(value, undo, mark)
            return
        cell = -1
        for cs, need in sorted(cur, key=lambda t: len(t[0])):
            if need > 0:
                for cand in cs:
                    if value[cand] < 0:
                        cell = cand
                        break
            if cell >= 0:
                break
        if cell < 0:  # everything decided (pins count too)
            record()
            _rollback(value, undo, mark)
            return
        for val in (1, 0):
            value[cell] = val
            dfs(cur)
            value[cell] = -1
        _rollback(value, undo, mark)

    if open_cons:
        start = _reduce(open_cons, value, undo)
        if start is None:
            raise Inconsistent("no assignment satisfies the component")
        dfs(start)
        if not hist:
            raise Inconsistent("no assignment satisfies the component")
        n_sol = sum(hist.values())
    else:
        record()
        n_sol = 1
    return cells, succ_by_k, hist, n_sol, nodes


def sample_group(group: list[Constraint], rng: np.random.Generator, max_samples: int):
    """Backtracking fallback for components too large to enumerate exhaustively.

    Same return shape as enumerate_group: (cells, succ_by_k, hist, n_sol, nodes).

    Plain rejection sampling dies on long chains (the acceptance rate is exponentially
    small), so this walks the constraints instead: pick the most constrained undecided
    constraint, place its mine count in the still-undecided cells uniformly at random, and
    backtrack on contradiction. That is a random walk over the *solution* space, so the
    second argument (`succ_by_k[j][m] = number of samples in which cell j is a mine and the
    component uses m mines`) estimates the same counts enumeration would give exactly.

    Because trials are uniform over solutions, probabilities computed from these counters
    are unbiased estimates of the exact ones; the search is deterministic per `rng`.
    """
    merged: dict[tuple[int, ...], int] = {}
    for c in group:
        key = tuple(sorted(c.cells))
        if key in merged and merged[key] != c.mines:
            raise Inconsistent("two constraints disagree on the same cells")
        merged[key] = c.mines
    cells = sorted({x for key in merged for x in key})
    idx = {c: i for i, c in enumerate(cells)}
    n = len(cells)
    cons = [(tuple(idx[x] for x in sorted(key)), k) for key, k in merged.items()]

    succ_by_k: list[dict[int, int]] = [dict() for _ in range(n)]
    hist: dict[int, int] = {}
    nodes = 0
    value = [-1] * n  # -1 undecided, 0 safe, 1 mine
    undo: list[int] = []
    found = 0
    deadline = time.perf_counter() + SAMPLE_SECONDS

    def reducecs(cur):
        """Decide what the constraints force under `value`; None when impossible."""
        work = cur
        while True:
            out = []
            f0: list[int] = []
            f1: list[int] = []
            for cs, k in work:
                und = [i for i in cs if value[i] < 0]
                need = k - sum(1 for i in cs if value[i] == 1)
                if need < 0 or need > len(und):
                    return None
                if und:
                    out.append((tuple(und), need))
                    if need == 0:
                        f0.extend(und)
                    elif need == len(und):
                        f1.extend(und)
            if not out:
                return []
            if not f0 and not f1:
                return out
            if set(f0) & set(f1):
                return None
            for i in f0:
                if value[i] == 1:
                    return None
                value[i] = 0
                undo.append(i)
            for i in f1:
                if value[i] == 0:
                    return None
                value[i] = 1
                undo.append(i)
            work = out

    def rollback(mark: int) -> None:
        while len(undo) > mark:
            value[undo.pop()] = -1

    def record() -> None:
        m = sum(1 for i in range(n) if value[i] == 1)
        hist[m] = hist.get(m, 0) + 1
        for i in range(n):
            if value[i] == 1:
                bucket = succ_by_k[i]
                bucket[m] = bucket.get(m, 0) + 1

    def walk(cur) -> None:
        nonlocal nodes, found
        nodes += 1
        if found >= max_samples or nodes > 2_000_000 or time.perf_counter() > deadline:
            return
        mark = len(undo)
        cur = reducecs(cur)
        if cur is None:
            rollback(mark)
            return
        if not cur:
            record()
            found += 1
            rollback(mark)
            return
        # most constrained open constraint: fewest undecided cells with need > 0
        target = None
        for cs, need in sorted(cur, key=lambda t: len(t[0])):
            if need > 0:
                target = (cs, need, [i for i in cs if value[i] < 0])
                break
        if target is None:
            record()
            found += 1
            rollback(mark)
            return
        cs, need, und = target
        need = min(need, len(und))
        mines = rng.choice(und, size=need, replace=False)
        chosen = set(int(x) for x in np.atleast_1d(mines))
        for i in und:
            value[i] = 1 if i in chosen else 0
            undo.append(i)
        walk(cur)
        rollback(mark)

    start = reducecs(cons)
    if start is None:
        raise Inconsistent("no assignment satisfies the component")
    if not start:
        record()
    else:
        warm = 0
        while found < max_samples and nodes < 2_000_000 and time.perf_counter() < deadline:
            walk(start)
            warm += 1
            if warm > 200_000:
                break
    if found == 0:
        raise Inconsistent("backtracking search found no solution")
    return cells, succ_by_k, hist, found, nodes


# --------------------------------------------------------------------------- #
# independent components
# --------------------------------------------------------------------------- #
def components(cons: list[Constraint]) -> list[list[Constraint]]:
    parent = list(range(len(cons)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner: dict[int, int] = {}
    for i, c in enumerate(cons):
        for cell in c.cells:
            if cell in owner:
                a, b = find(i), find(owner[cell])
                if a != b:
                    parent[a] = b
            else:
                owner[cell] = i
    groups: dict[int, list[Constraint]] = {}
    for i, c in enumerate(cons):
        groups.setdefault(find(i), []).append(c)
    return list(groups.values())


def count_group_dp(group: list[Constraint], cols: int):
    """Exact histograms with shared suffixes for large, offline frontier counts.

    At a fixed sweep position two prefixes with identical outstanding clue counts
    admit the same completions. Keep their multiplicities by total mines. A backward
    pass joins each mine edge's prefix and suffix histograms to get its marginal.
    """
    if any(not 0 <= clue.mines <= len(clue.cells) for clue in group):
        raise Inconsistent("invalid mine count for a component clue")
    group = list(dict.fromkeys(c for c in group if c.cells))
    if not group:
        return [], [], {0: 1}, 1, 0
    cells = sorted({x for clue in group for x in clue.cells})
    indices = {cell: i for i, cell in enumerate(cells)}
    column_order = sorted(cells, key=lambda c: (c % cols, c // cols))

    def sweep_cost(order):
        positions = {cell: i for i, cell in enumerate(order)}
        cuts = [0] * len(order)
        for clue in group:
            places = [positions[c] for c in clue.cells]
            for i in range(min(places), max(places)):
                cuts[i] += 1
        return max(cuts), sum(cuts)

    order = min((cells, column_order), key=sweep_cost)
    touches = {c: [] for c in cells}
    places = {c: i for i, c in enumerate(order)}
    for j, clue in enumerate(group):
        for remaining, cell in enumerate(sorted(clue.cells, key=places.get, reverse=True)):
            touches[cell].append((j, remaining))

    root = tuple(clue.mines for clue in group)
    prefixes = {root: {0: 1}}
    layers = []
    states = 0
    for cell in order:
        edges = {}
        following = {}
        for residual, hist in prefixes.items():
            children = []
            for bit in (0, 1):
                if any(not 0 <= residual[j] - bit <= left for j, left in touches[cell]):
                    continue
                child = list(residual)
                for j, _ in touches[cell]:
                    child[j] -= bit
                key = tuple(child)
                children.append((bit, key))
                target = following.setdefault(key, {})
                for k, n in hist.items():
                    target[k + bit] = target.get(k + bit, 0) + n
            edges[residual] = children
        states += len(edges)
        layers.append((prefixes, edges))
        prefixes = following
    terminal = (0,) * len(group)
    if terminal not in prefixes:
        raise Inconsistent("no assignment satisfies the component")
    total = prefixes[terminal]
    succ = [{} for _ in cells]
    suffixes = {terminal: {0: 1}}
    for cell, (prefix, edges) in zip(reversed(order), reversed(layers)):
        before = {}
        hits = succ[indices[cell]]
        for residual, children in edges.items():
            tail = {}
            for bit, key in children:
                for k, n in suffixes.get(key, {}).items():
                    tail[k + bit] = tail.get(k + bit, 0) + n
                    if bit:
                        for j, m in prefix[residual].items():
                            s = j + 1 + k
                            hits[s] = hits.get(s, 0) + m * n
            before[residual] = tail
        suffixes = before
    return cells, succ, total, sum(total.values()), states


def comps_hist(comps: list[dict], budget: int) -> dict[int, int]:
    """Convolve every component's mine histogram into one distribution."""
    out: dict[int, int] = {0: 1}
    for comp in comps:
        out = convolve(out, comp["hist"], budget)
    return out


def convolve(dist: dict[int, int], hist: dict[int, int], budget: int) -> dict[int, int]:
    out: dict[int, int] = {}
    for k, w in dist.items():
        for m, hw in hist.items():
            if k + m <= budget:
                out[k + m] = out.get(k + m, 0) + w * hw
    return out


class Solver:
    """Exact-by-construction, latency-bounded."""

    def __init__(self, seed: int = 0, max_nodes: int | None = 200_000, max_samples: int = 3000,
                 *, offsets=NEIGHBOURS):
        self.offsets = offsets
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.max_nodes = max_nodes
        self.max_samples = max_samples
        self.last_stats: dict = {}

    def _propagate(self, board: Board, stats: dict) -> Move | None:
        cols = board.cols
        if not board.covered().any():
            self.last_stats = {"kind": "done"}
            return Move("done", (), 0.0, "no covered cells")

        raw, frontier, outside = raw_constraints(board, offsets=self.offsets)
        safe, mines = certain_moves(raw)
        stats.update(raw_cons=len(raw), frontier=len(frontier), outside=len(outside))
        if safe:
            stats.update(kind="certain_open", decided=len(safe))
            self.last_stats = stats
            return Move(
                "open",
                tuple(sorted((x // cols, x % cols) for x in safe)),
                0.0,
                f"propagation: {len(safe)} cell(s) safe",
            )
        if mines:
            stats.update(kind="certain_flag", decided=len(mines))
            self.last_stats = stats
            return Move(
                "flag",
                tuple(sorted((x // cols, x % cols) for x in mines)),
                1.0,
                f"propagation: {len(mines)} cell(s) are mines",
            )

        return None

    def _global_move(self, stats: dict, safe_now: set[Cell], certain_mine: set[Cell]) -> Move | None:
        if safe_now:
            stats.update(kind="certain_open_full", decided=len(safe_now))
            self.last_stats = stats
            return Move("open", tuple(sorted(safe_now)), 0.0,
                        f"probabilities: {len(safe_now)} cell(s) provably safe")
        if certain_mine:
            stats.update(kind="certain_flag_full", decided=len(certain_mine))
            self.last_stats = stats
            return Move("flag", tuple(sorted(certain_mine)), 1.0,
                        f"probabilities: {len(certain_mine)} cell(s) provably mines")

        return None

    def deduce(self, board: Board, *, belief: dict[Cell, float] | None = None) -> Move | None:
        """Return a proven move or done. None certifies a guess. A capped search may raise BudgetExceeded."""
        if belief is not None:
            belief.clear()
        stats: dict = {}
        move = self._propagate(board, stats)
        if move is not None:
            return move
        counts: dict[Cell, int] = {}
        probs, _ = self.measure(board, stats, mine_counts=counts, exact_only=True)
        stats["kind"] = "guess"
        self.last_stats = stats
        move = self._global_move(stats, {rc for rc, n in counts.items() if n == 0},
                                 {rc for rc, n in counts.items() if n == stats["solutions"]})
        if move is None and belief is not None:
            belief.update(probs)
        return move

    def decide(self, board: Board, *, guess_ranker: GuessRanker | None = None,
               exact_guess_counts: dict[Cell, int] | None = None) -> Move:
        """Play normally; expose counts only on an exactly measured guess if requested."""
        if exact_guess_counts is not None:
            exact_guess_counts.clear()
        stats: dict = {}
        move = self._propagate(board, stats)
        if move is not None:
            return move
        counts: dict[Cell, int] | None = (
            {} if guess_ranker is not None or exact_guess_counts is not None else None)
        probs, _ = self.measure(board, stats, mine_counts=counts)
        move = self._global_move(stats, {rc for rc, p in probs.items() if p <= 1e-12},
                                 {rc for rc, p in probs.items() if p >= 1.0 - 1e-12})
        if move is not None:
            return move
        best = min(probs, key=lambda rc: (probs[rc], -self._adjacent_numbers(board, rc)))
        reason = f"guess: min p={probs[best]:.4f}"
        if exact_guess_counts is not None and counts is not None and not stats["degraded"]:
            exact_guess_counts.update(counts)
        if guess_ranker is not None and counts is not None and not stats["degraded"]:
            pick = guess_ranker(board, best, counts, stats["solutions"])
            if pick != best and counts.get(pick) == counts[best]:
                best = pick
                reason += " | lookahead broke exact-risk tie"
        p = probs[best]
        stats.update(kind="guess", guess_p=p)
        self.last_stats = stats
        return Move("guess", (best,), float(p), reason)

    # -- probability engine ------------------------------------------------ #
    def measure(self, board: Board, stats: dict | None = None, *,
                mine_counts: dict[Cell, int] | None = None, exact_only: bool = False,
                component_cache: ComponentCache | None = None):
        """Per-cell mine probability over the visible board; returns (probs, comps).

        When supplied, ``mine_counts`` receives integer numerators over ``stats['solutions']``.
        These are exact only if ``stats['degraded']`` is false. ``exact_only`` additionally
        validates closed clues and lets BudgetExceeded propagate instead of sampling.

        Method (no heuristics): every arrangement of mines consistent with the revealed
        numbers and the remaining mine count is a "world". A cell's probability is the
        fraction of worlds containing a mine there. Worlds are grouped by the per-component
        mine count k, so the sum over worlds collapses to
            sum_k  ways(comp uses k) * ways(rest of board fits | k) * P(cell | comp, k)
        where the last factor is exact inside the component (enumerated).
        """
        rows, cols = board.rows, board.cols
        budget = int(board.mines_left)
        if budget < 0:
            raise Inconsistent(f"mine counter is negative ({budget})")
        stats = {} if stats is None else stats
        raw, _, outside = raw_constraints(board, strict=exact_only, offsets=self.offsets)

        comps = []
        degraded = False
        for g in components(raw):
            try:
                if self.max_nodes is None and component_cache is None:
                    try:
                        measured = enumerate_group(g, 1500)
                    except BudgetExceeded:
                        measured = count_group_dp(g, cols)
                elif component_cache is None:
                    measured = enumerate_group(g, self.max_nodes)
                else:
                    key = tuple(sorted(set(g), key=lambda c: (c.cells, c.mines)))
                    # Only exhaustive counts are cached; the original traversal sets
                    # the cost. Do not seed this cache with approximate sample counts.
                    measured = component_cache.get(key)
                    if measured is None:
                        measured = enumerate_group(g, self.max_nodes)
                        component_cache[key] = measured
                    elif self.max_nodes is not None and measured[-1] > self.max_nodes:
                        raise BudgetExceeded
                cells, succ_by_k, hist, n_sol, _ = measured
            except BudgetExceeded:
                if exact_only:
                    raise
                try:
                    cells, succ_by_k, hist, n_sol, _ = sample_group(
                        g, self.rng, self.max_samples)
                except (Inconsistent, BudgetExceeded):
                    raise Inconsistent("frontier too large to enumerate or sample")
                degraded = True
            if not hist:  # nothing open: one solution, no mine count information
                hist = {sum(1 for s in succ_by_k if s): 1}
                n_sol = 1
            comps.append({"cells": cells, "hist": hist, "n_sol": n_sol,
                          "succ_of": {cells[j]: dict(succ_by_k[j])
                                      for j in range(len(cells))}})
        stats.update(components=len(comps), degraded=degraded)

        listed = {x for comp in comps for x in comp["cells"]}
        free_cells = sorted(outside - listed)
        free_n = len(free_cells)

        # Distribute mines between the components: dist[k] = ways to place exactly k mines
        # on the whole frontier; the free cells take the remainder with a binomial factor.
        dist: dict[int, int] = {0: 1}
        for comp in comps:
            dist = convolve(dist, comp["hist"], budget)

        def fit(used: int) -> int:
            """How many complete boards fit if `used` mines sit on the enumerated frontier."""
            left = budget - used
            if free_n:
                return comb(free_n, left) if 0 <= left <= free_n else 0
            return 1 if left == 0 else 0

        total_w = sum(w * fit(k) for k, w in dist.items())
        if total_w == 0:
            raise Inconsistent(f"mine counter {budget} is impossible for this visible board")
        stats["solutions"] = total_w

        probs: dict[tuple[int, int], float] = {}
        for idx, comp in enumerate(comps):
            others: dict[int, int] = {0: 1}
            for j, other in enumerate(comps):
                if j != idx:
                    others = convolve(others, other["hist"], budget)
            for cell, by_k in comp["succ_of"].items():
                # weight each k bucket by (ways this comp uses k) * (ways the rest fits)
                mine_mass = 0.0
                exact_mass = 0
                for k, ways in comp["hist"].items():
                    if not ways:
                        continue
                    rest = 0
                    for rk, rw in others.items():
                        rest += rw * fit(k + rk)
                    if rest:
                        worlds = rest * by_k.get(k, 0)
                        mine_mass += worlds
                        if mine_counts is not None:
                            exact_mass += worlds
                rc = (cell // cols, cell % cols)
                probs[rc] = mine_mass / total_w
                if mine_counts is not None:
                    mine_counts[rc] = exact_mass

        if free_n:
            mass = 0
            for k, ways in dist.items():
                left = budget - k
                if 0 <= left <= free_n:
                    mass += ways * comb(free_n, left) * left
            pf = mass / total_w / free_n
            # Symmetry: mass sums the mines over all free cells and all worlds.
            each_mass = mass // free_n
            for cell in free_cells:
                rc = (cell // cols, cell % cols)
                probs[rc] = pf
                if mine_counts is not None:
                    mine_counts[rc] = each_mass

        for r in range(rows):
            for c in range(cols):
                if int(board.cells[r, c]) in (COVERED, QUESTION):
                    probs.setdefault((r, c), 0.0)
                    if mine_counts is not None:
                        mine_counts.setdefault((r, c), 0)
        return probs, comps

    # -- helpers ----------------------------------------------------------- #
    def _adjacent_numbers(self, board: Board, rc) -> int:
        """Tiebreak only: cells touching more revealed numbers teach us more."""
        rows, cols = board.rows, board.cols
        return sum(1 for rr, cc in _nbrs(rows, cols, *rc) if int(board.cells[rr, cc]) >= 0)
