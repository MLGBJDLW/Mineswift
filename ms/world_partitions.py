"""Exact terminal scores on enumerated layouts, memoized by compatible layouts."""
from .endgame import _observations
from .solver import BudgetExceeded


class WorldPartitions:
    def __init__(self, board, hidden, worlds, work):
        rows = _observations(board, hidden, worlds)
        self.all = (1 << len(worlds)) - 1
        self.groups = []
        self.safe = []
        self.memo = {}
        self.work = work
        for cell in range(len(hidden)):
            parts = {}
            safe = 0
            for w, (mines, row) in enumerate(zip(worlds, rows)):
                if not (mines >> cell) & 1:
                    bit = 1 << w
                    key = row[cell]
                    parts[key] = parts.get(key, 0) | bit
                    safe |= bit
            self.groups.append(tuple(parts.values()))
            self.safe.append(safe)

    def spend(self, work):
        self.work -= work
        if self.work < 0:
            raise BudgetExceeded

    def click(self, cell, feasible):
        groups = self.groups[cell]
        self.spend(len(groups))
        return sum(self.value(child) for group in groups if (child := feasible & group))

    def value(self, feasible):
        if feasible.bit_count() == 1:
            return 1
        if feasible not in self.memo:
            self.memo[feasible] = self.best(feasible)
        return self.memo[feasible]

    def best(self, feasible):
        # Full initial flood observations are reusable: knowing the new and old
        # digits lets us reconstruct the entire flood from the clicked square.
        # Previously opened cells therefore give constant observations here.
        self.spend(len(self.safe))
        risky = []
        for cell, safe in enumerate(self.safe):
            survivors = feasible & safe
            if survivors == feasible:
                groups = self.groups[cell]
                self.spend(len(groups))
                if any((feasible & group) not in (0, feasible) for group in groups):
                    return self.click(cell, feasible)
            elif survivors:
                risky.append((survivors.bit_count(), cell))
        # With distinct layouts at least one unresolved mine differs. Committing
        # to a compatible layout always wins at least one of these worlds.
        wins = 1
        for upper, cell in sorted(risky, reverse=True):
            if upper <= wins:
                break
            wins = max(wins, self.click(cell, feasible))
        return wins

    def compare(self, incumbent, order):
        """Switch only on a completely evaluated, strictly better terminal score."""
        try:
            original = self.click(incumbent, self.all)
        except BudgetExceeded:
            return None
        best, wins = incumbent, original
        for cell in order:
            if cell == incumbent or self.safe[cell].bit_count() <= wins:
                continue
            try:
                score = self.click(cell, self.all)
            except BudgetExceeded:
                return best, original, wins, False
            if score > wins:
                best, wins = cell, score
        return best, original, wins, True
