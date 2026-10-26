from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.params import ROOT

RUNS = ROOT / "runs"

CHARTS = [
    ("rollout/passes_per_ep", "passes_per_ep", "Passes per episode (last 200 episodes)"),
    ("rollout/passes_per_min", "passes_per_min", "Passes per minute (last 200 episodes)"),
    ("curriculum/level", "curriculum_level", "Curriculum level"),
    ("rollout/crash_rate", "crash_rate", "Crash rate (last 200 episodes)"),
    ("rollout/ep_rew_mean", "episode_reward", "Episode reward"),
    ("rollout/ep_len_mean", "episode_length", "Episode length (steps)"),
    ("train/critic_loss", "critic_loss", "Critic loss"),
    ("train/actor_loss", "actor_loss", "Actor loss"),
    ("train/ent_coef", "entropy_coef", "Entropy coefficient"),
]

RAW = "#b7d3f6"
SMOOTH = "#2a78d6"
INK = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
SMOOTHING = 0.9


def find_log_dir(name: str, runs: Path | None = None) -> Path:
    runs = runs or RUNS
    pattern = re.compile(rf"^{re.escape(name)}_(\d+)$")
    found = [(int(m.group(1)), p) for p in runs.iterdir() if (m := pattern.match(p.name))]
    if not found:
        raise FileNotFoundError(f"no TensorBoard log for {name} in {runs}")
    return max(found)[1]


def read_scalars(log_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    events = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    events.Reload()
    return {tag: (np.array([e.step for e in events.Scalars(tag)]),
                  np.array([e.value for e in events.Scalars(tag)]))
            for tag in events.Tags()["scalars"]}


def smooth(values: np.ndarray, weight: float = SMOOTHING) -> np.ndarray:
    out, last = np.empty(len(values)), values[0]
    for i, v in enumerate(values):
        last = weight * last + (1 - weight) * v
        out[i] = last
    return out


def _draw(ax, steps, values, title, tag):
    ax.plot(steps, values, color=RAW, linewidth=1)
    ax.plot(steps, smooth(values), color=SMOOTH, linewidth=2)
    ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9, length=0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e6:g}M" if x >= 1e6 else f"{x / 1e3:g}k"))
    ax.set_xlabel("environment steps", color=INK, fontsize=9)
    if tag == "rollout/crash_rate":
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))


def save_plots(name: str, log_dir: Path | None = None) -> Path:
    scalars = read_scalars(log_dir or find_log_dir(name))
    charts = [c for c in CHARTS if c[0] in scalars and len(scalars[c[0]][0])]
    out = RUNS / name / "plots"
    out.mkdir(parents=True, exist_ok=True)
    caption = f"{name}   light: every logged point, dark: smoothed"

    for tag, file, title in charts:
        fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
        _draw(ax, *scalars[tag], title, tag)
        fig.text(0.01, 0.01, caption, color=INK, fontsize=8)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(out / f"{file}.png", dpi=150)
        plt.close(fig)

    rows = (len(charts) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3.4 * rows), facecolor=SURFACE, squeeze=False)
    for ax, (tag, _, title) in zip(axes.flat, charts):
        _draw(ax, *scalars[tag], title, tag)
    for ax in axes.flat[len(charts):]:
        ax.set_visible(False)
    fig.text(0.01, 0.005, caption, color=INK, fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out / "overview.png", dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+", help="run names, as passed to train.py --name")
    for name in parser.parse_args().names:
        print(f"saved {save_plots(name)}")
