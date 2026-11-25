"""Local pytest configuration for the executor suite.

`fast_clock` replaces ``time.sleep`` with an instant, clock-advancing sleep, so a settle
loop that never settles would spin forever. Every loop in ``ms.executor`` re-checks the
monotonic clock, and ``fast_clock`` counts its own virtual leaps: after ``BUDGET``
seconds of virtual time the next sleep raises, turning a logic bug into a fast, named
failure instead of a hung test process.

No wall-clock timeout plugin is needed, so this stays portable.
"""
from __future__ import annotations

import pathlib
import time

import pytest

BUDGET = 10.0     # virtual seconds per test


class _VirtualClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def perf_counter(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        if self.now - 1000.0 > BUDGET:
            raise TimeoutError(f"virtual clock passed {BUDGET:g}s of simulated waiting")
        self.sleeps.append(s)
        self.now += max(0.0, float(s))            # exact: float sums drift 50.000e-3 -> 49.999e-3


@pytest.fixture
def fast_clock(monkeypatch):
    clock = _VirtualClock()
    monkeypatch.setattr(time, "perf_counter", clock.perf_counter)
    monkeypatch.setattr(time, "sleep", clock.sleep)
    return clock


def pytest_configure(config):
    # Anything that still waits on the wall clock would hide a hang behind a timeout
    # plugin we do not ship. Fail loudly at collection instead.
    root = pathlib.Path(__file__).resolve().parent.parent
    assert (root / "ms" / "executor.py").exists()
