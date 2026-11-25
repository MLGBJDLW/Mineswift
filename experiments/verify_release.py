"""Verify archived per-game outcomes, seed protocols, and checkpoint identity."""

import json
from dataclasses import asdict
from hashlib import sha256
from math import isclose

import numpy as np

from ms.board import Board
from ms.engine import EXPERT
from ms.opening import first_cell
from ms.play_web import ROOT, WEIGHTS, load_policy
from .release_eval import summary

FROZEN = "505985d5d667df43bc9cd0a6de5475d9fe6108f600167ee30b1a3560a80adcca"
SOURCES = (
    ("belief_rl_holdout_1024.json", 530000, "belief_rollout"),
    ("web_guess_holdout_1024.json", 650000, "live"),
    ("release_eval_830000.json", 830000, "live"),
)


def verify():
    if sha256(WEIGHTS.read_bytes()).hexdigest() != FROZEN:
        raise ValueError("release checkpoint does not match the evaluated policy")
    combined = []
    for filename, start, policy in SOURCES:
        data = json.loads((ROOT / "reports" / filename).read_text(encoding="utf-8"))
        hashes = data.get("checkpoints", {"belief_rl.pt": data.get("weights_sha256")})
        for name, digest in hashes.items():
            if sha256((ROOT / "models" / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"{filename}: checkpoint mismatch for {name}")
        rows = data["boards"]
        if (
            data["seed_base"] != start
            or data["games"] != 1024
            or data["opening"] != "random"
            or data["cfg"] != asdict(EXPERT)
            or len(rows) != 1024
        ):
            raise ValueError(f"{filename}: wrong protocol")
        batch = []
        for seed, row in enumerate(rows, start):
            if row["seed"] != seed or row["first_cell"] != first_cell(
                EXPERT, seed, "random"
            ):
                raise ValueError(f"{filename}: mismatched board or opening at {seed}")
            if any(
                row[key] not in {"win", "guess", "logic", "stuck"}
                for key in (policy, "endgame")
            ):
                raise ValueError(f"{filename}: unknown outcome at {seed}")
            batch.append({"seed": seed, "live": row[policy], "endgame": row["endgame"]})
        totals = summary(batch)
        for name, key in (("live", policy), ("endgame", "endgame")):
            if "policies" in data:
                recorded = data["policies"][key]["wins"]
            elif "outcomes" in data:
                recorded = data["outcomes"][key]["win"]
            else:
                recorded = data[key]["wins"]
            if totals[name]["wins"] != recorded:
                raise ValueError(f"{filename}: per-game wins disagree for {name}")
        combined.extend(batch)
        print(filename, json.dumps(totals))
    if len({r["seed"] for r in combined}) != len(combined):
        raise ValueError("overlapping test seeds")
    return summary(combined)


def verify_default_endgames():
    cases = json.loads((ROOT / "reports/verified_guesses_cases.json").read_bytes())["cases"]
    if len(cases) != 11 or len({case["seed"] for case in cases}) != 11:
        raise ValueError("expected 11 distinct endgame regression cases")
    for case in cases:
        seed = case["seed"]
        cells = np.asarray(case["cells"], dtype=np.int8)
        board = Board(*cells.shape, cells, case["remaining_mines"])
        policy = load_policy(seed)
        move = policy.act(board)
        expected = case["change"]
        cell = divmod(expected["new_cell"], board.cols)
        if move.kind != "guess" or move.cells != (cell,):
            raise ValueError(f"seed {seed}: expected guess at {cell}, got {move}")
        details = policy.last_guess
        if (details.get("verified_worlds") != case["worlds"]
                or details.get("optimal_wins") != expected["new_wins"]
                or details.get("mine_risk") != move.p_mine
                or not isclose(move.p_mine, expected["new_risk"], abs_tol=1e-7)):
            raise ValueError(f"seed {seed}: endgame review disagrees with archived result: {details}")
    return len(cases)


if __name__ == "__main__":
    print("Combined", json.dumps(verify()))
    print(f"Default desktop policy matched {verify_default_endgames()} archived endgames.")
