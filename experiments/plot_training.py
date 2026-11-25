"""把已保存的训练批次和同盘验证结果画成中文图表。"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator
import numpy as np


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def batches(axis, x, values, color):
    values = np.asarray(values, dtype=float)
    axis.plot(x, values, color=color, alpha=.33, linewidth=1)
    if len(values) >= 8:
        axis.plot(x[7:], np.convolve(values, np.ones(8)/8, mode="valid"),
                  color=color, linewidth=2)
    axis.set_xlim(0, x[-1])
    axis.xaxis.set_major_locator(MaxNLocator(5, integer=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", default="reports/belief_rl_train_2048.json")
    parser.add_argument("--previous", default="reports/belief_selection_710000.json")
    parser.add_argument("--study", default="reports/belief_selection_1100000.json")
    parser.add_argument("--out", type=Path, default=Path("reports/training_progress.png"))
    args = parser.parse_args()
    history, previous, new = map(load, (args.initial, args.previous, args.study))
    candidates = ["Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", "SimHei", "PingFang SC"]
    plt.rcParams["font.sans-serif"] = candidates + ["DejaVu Sans"]
    if not any(f.name in candidates for f in font_manager.fontManager.ttflist):
        parser.error("请先安装支持中文的字体，例如 Noto Sans CJK SC")
    fig = plt.figure(figsize=(13, 8.8), layout="constrained")
    grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.02])
    axes = []
    for col, (label, run) in enumerate(zip(("从零训练", "上次续训", "本次续训"), (history, previous, new))):
        log = run["batches"]
        group = run.get("plan", run)["group"]
        counts = np.array([group*b["layouts"] for b in log])
        x = np.cumsum(counts)
        win_ax = fig.add_subplot(grid[0, col])
        axes.append(win_ax)
        batches(win_ax, x, [100*b["wins"]/n for b,n in zip(log, counts)], "#00776d")
        win_ax.set_title(f"{label}  ·  {x[-1]:,} 盘", loc="left", fontweight="bold")
        win_ax.set_ylim(0, 100)
        win_ax.set_ylabel("每批训练胜率（%）" if col == 0 else "")
        metric_ax = fig.add_subplot(grid[1, col], sharex=win_ax)
        axes.append(metric_ax)
        if col == 0:
            batches(metric_ax, x, [b["last_grad_norm"] for b in log], "#626d80")
            metric_ax.set_title("梯度范数（该次未记录 loss）", loc="left", fontsize=11)
        else:
            batches(metric_ax, x, [b["epochs"][-1]["loss"] if b["epochs"] else np.nan for b in log], "#ad541e")
            metric_ax.axhline(0, color="#695b53", linewidth=.7)
            metric_ax.set_title("策略代理损失（每批最后一次更新）", loc="left", fontsize=11)
        metric_ax.set_xlabel("本次运行累计训练盘数")
    validation = new["validation"]["wins"]
    n = new["plan"]["validation"][1] - new["plan"]["validation"][0]
    checkpoints = new["plan"]["checkpoints"]
    keys = [k for k in ["endgame", "live", *map(str, checkpoints)] if k in validation]
    labels = {"live":"现版模型", "endgame":"Endgame 基线"}
    ax = fig.add_subplot(grid[2, :])
    axes.append(ax)
    for pos, key in enumerate(keys):
        value = validation[key]*100/n
        ax.barh(pos, value, color=("#425673" if key in labels else "#128278"), height=.57)
        ax.text(value+.6, pos, f"{value:.2f}%  ·  {validation[key]}/{n}", va="center", fontsize=10)
    names = [labels[k] if k in labels else f"追加 {int(k) * new['plan']['group']:,} 盘" for k in keys]
    ax.set_yticks(range(len(keys)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("专家盘胜率（%）")
    ax.set_title(f"独立验证  ·  同一组 {n:,} 张专家棋盘，相同随机首点", loc="left", fontweight="bold")
    for a in axes:
        a.spines[["right", "top"]].set_visible(False)
        a.grid(axis="x" if a is ax else "y", alpha=.15)
        a.set_axisbelow(True)
    fig.suptitle("Mineswift  /  训练与验证\n每张新雷图采样四盘；细线为每批实测，粗线为最近八批均值。两次续训均从现版权重出发。",
                 fontsize=14, ha="left", x=.01)
    fig.savefig(args.out, dpi=145, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
