"""See how a policy flies: replay episodes and draw each flight path.

Run: python -m src.replay --model runs/<name>/model.zip
     python -m src.replay --model runs/<name>/checkpoints/model_1000000_steps.zip
     python -m src.replay --baseline
Options: --episodes 6, --level 1.0, --randomise, --out <dir>,
         --no-hanger for models trained before the hanger was modelled

Uses the evaluation layouts (evaluate.EVAL_SEED), so episode k here is episode k
there. Writes one PNG per episode, two views each:
    from above   the cage, the hoop and the path, light (start) to dark (end)
    face-on      the path as seen from the hoop's front, in front and behind in
                 two colours, with the hanger: this shows which way it goes round
Prints how each episode ended and which way every lap went round.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # the HPC has no display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse, Rectangle

from src.evaluate import EVAL_SEED, load_policy
from src.baseline import Lapper
from src.plots import GRID, INK, SURFACE
from src import contract
from src.sim import HoopEnv

TIME = LinearSegmentedColormap.from_list("time", ["#b7d3f6", "#0d366b"])  # blue 150 -> 700
FRONT, BACK = "#2a78d6", "#eb6834"   # categorical slots 1 and 2
HOOP = "#0b0b0b"
CRASH = "#e34948"                    # critical; always labelled


def fly(policy, env: HoopEnv, seed: int) -> dict:
    """One episode: positions, pass points, routes and how it ended."""
    obs, info = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    if hasattr(policy, "bind"):   # a privileged teacher reads the sim's true state
        policy.bind(env)
    path, passes, routes, done = [info["drone_pos"]], [], [], False
    while not done:
        obs, _, terminated, truncated, info = env.step(policy(obs))
        done = terminated or truncated
        path.append(info["drone_pos"])
        if info["outcome"] == "passed" or (done and info["passes"] > len(passes)):
            passes.append(info["drone_pos"])
        if info["route"]:
            routes.append(info["route"])
    return {"path": np.array(path), "passes": np.array(passes).reshape(-1, 3), "routes": routes,
            "outcome": info["outcome"], "seconds": info["steps"] * env.drone.dt}


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=10, color=HOOP)
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal")
    ax.tick_params(colors=INK, labelsize=8, length=0)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def _timeline(ax, xy):
    t = np.linspace(0, 1, len(xy) - 1)
    segments = np.stack([xy[:-1], xy[1:]], axis=1)
    ax.add_collection(LineCollection(segments, colors=TIME(t), linewidths=2))


def draw(env: HoopEnv, run: dict, title: str, path: Path):
    cage, m = env.cage, env.cage.margin
    c, n, h = env.hoop_centre, env.hoop_normal, env._axis_h
    radius_h, radius_v = np.linalg.norm(env._hoop_h), np.linalg.norm(env._hoop_v)
    xyz = run["path"]

    fig, (top, face) = plt.subplots(1, 2, figsize=(11, 5.4), facecolor=SURFACE)

    # From above.
    top.add_patch(Rectangle((0, 0), cage.length, cage.width, fill=False, ec=INK, lw=1))
    top.add_patch(Rectangle((m, m), cage.length - 2 * m, cage.width - 2 * m, fill=False,
                            ec=INK, lw=0.8, ls="--"))
    rim = np.array([c + radius_h * h, c - radius_h * h])
    top.plot(rim[:, 0], rim[:, 1], color=HOOP, lw=4, solid_capstyle="butt")
    top.annotate("", xy=c[:2] - 0.25 * n[:2], xytext=c[:2], arrowprops=dict(arrowstyle="->", color=INK))
    top.text(*(c[:2] - 0.3 * n[:2]), "front", color=INK, fontsize=8, ha="center", va="center")
    _timeline(top, xyz[:, :2])
    top.plot(*xyz[0, :2], "o", color=TIME(0.0), mec=INK, ms=8)
    top.text(xyz[0, 0], xyz[0, 1] + 0.07, "start", color=INK, fontsize=8, ha="center")
    crashed = run["outcome"] in contract.CRASHES
    if crashed:
        top.plot(*xyz[-1, :2], "X", color=CRASH, mec=HOOP, ms=11)
    top.set_xlim(-0.05, cage.length + 0.05)
    top.set_ylim(-0.05, cage.width + 0.05)
    _style(top, "From above   (light = early, dark = late; dashed = layout margin)")

    # Face-on, looking from the front: +axis_h is the viewer's left, so negate it.
    rel = xyz - c
    across, up, depth = -(rel @ h), xyz[:, 2], rel @ n
    for sign, colour, label in ((-1, FRONT, "in front of the hoop"), (1, BACK, "behind the hoop")):
        mask = np.sign(depth) == sign   # NaN breaks the line where the other side takes over
        face.plot(np.where(mask, across, np.nan), np.where(mask, up, np.nan),
                  color=colour, lw=1.6, label=label)
    face.add_patch(Ellipse((0, c[2]), 2 * radius_h, 2 * radius_v, fill=False, ec=HOOP, lw=4))
    face.add_patch(Rectangle((-env.hanger_half_width, c[2] + radius_v),
                             2 * env.hanger_half_width, cage.height - c[2] - radius_v,
                             fc=GRID, ec=INK, lw=0.8, hatch="///", label="hanger"))
    face.axhline(cage.min_altitude, color=INK, lw=0.8, ls="--")
    face.axhline(cage.height - m, color=INK, lw=0.8, ls="--")
    if len(run["passes"]):
        p = run["passes"] - c
        face.plot(-(p @ h), run["passes"][:, 2], "o", color=HOOP, ms=4, label="pass")
    if crashed:
        face.plot(across[-1], up[-1], "X", color=CRASH, mec=HOOP, ms=11, label=run["outcome"])
    reach = max(1.2, np.abs(across).max() + 0.1)
    face.set_xlim(-reach, reach)
    face.set_ylim(0, cage.height)
    face.legend(loc="upper left", fontsize=8, frameon=False)
    _style(face, "Face-on, seen from the front (dashed = height limits)")
    face.set_xlabel("metres across the hoop", color=INK, fontsize=8)
    face.set_ylabel("height, m", color=INK, fontsize=8)

    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=HOOP)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--baseline", action="store_true")
    group.add_argument("--model", help="a model.zip or a checkpoint")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--level", type=float, default=1.0)
    parser.add_argument("--randomise", action="store_true")
    parser.add_argument("--no-hanger", action="store_true",
                        help="leave the hanger out, for models trained before it existed")
    parser.add_argument("--out", help="default: replay_<model> next to the model")
    args = parser.parse_args()

    if args.baseline:
        policy, out = Lapper(), Path("runs/replay_baseline")
    else:
        model = Path(args.model)
        policy, out = load_policy(str(model)), model.parent / f"replay_{model.stem}"
    out = Path(args.out) if args.out else out
    out.mkdir(parents=True, exist_ok=True)

    env = HoopEnv(level=args.level, randomise=args.randomise, hanger=not args.no_hanger)
    all_routes, endings = Counter(), Counter()
    for i in range(args.episodes):
        run = fly(policy, env, EVAL_SEED + i)
        routes = Counter(run["routes"])
        all_routes.update(routes)
        endings[run["outcome"]] += 1
        way = ", ".join(f"{k} {v}" for k, v in routes.most_common()) or "never went round"
        title = (f"Episode {i}: {len(run['passes'])} passes, {run['outcome']} at "
                 f"{run['seconds']:.1f} s.  Went round: {way}")
        draw(env, run, title, out / f"episode_{i}.png")
        print(title)
    print(f"\nall episodes  went round: {dict(all_routes.most_common())}  ended: {dict(endings)}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
