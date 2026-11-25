"""Mineswift - a screen-capable minesweeper agent.

Layer map (each layer only talks to the next through Board / Move):

    ms.board       the data contract (Board, Move lives in ms.solver)
    ms.engine      rules + simulator; ground truth for offline evaluation
    ms.solver      constraint propagation, exact enumeration, exact probabilities
    ms.policy      decision policies, including the learned move scorer (the "AI")
    ms.eval        large-N offline evaluation with paired comparison and CIs
    ms.dataset     self-play data generation for training the scorer
    ms.perception  screenshot -> Board
    ms.chording    solver `open` -> fewest mouse actions (both-button clicks)
    ms.executor    Action -> real mouse, settle + re-read, per-site chord probe
    ms.agent       (next) state machine + emergency stop
"""

__version__ = "0.1.0"
