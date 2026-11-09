"""How far a Cosmos Transfer1 output drifts from its source within a 121-frame segment: per-frame median table
brightness of the output minus the source, averaged over a seeded sample of segments, for one or more batches.

  python.sh scripts/cosmos/cosmos_drift_profile.py --batch "policy 4 batch=$C/prep200_v2:$C/out200_v2" \
      --out results/figures/cosmos_drift_profile.png

Transfer1 copies the conditioning frame, so frame 0 always matches; the curve shows what the model does afterwards.
The table band is rows 40-55 % of the frame (between the robot base and the cubes' usual area).
"""
import argparse
import glob
import json
import random

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, INK, INK2, MUTED, GRID, AXIS, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def table_grey(path):
    cap, out = cv2.VideoCapture(path), []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        h = g.shape[0]
        out.append(float(np.median(g[int(h * 0.4):int(h * 0.55)])))
    cap.release()
    return np.array(out)


def profile(prep, cosmos_out, num, seed):
    rows = [json.loads(l) for l in open(f"{prep}/batch.jsonl")]
    random.seed(seed)
    diffs = []
    for i in sorted(random.sample(range(len(rows)), min(num, len(rows)))):
        vids = glob.glob(f"{cosmos_out}/video_{i}/*.mp4")
        if not vids:
            continue
        a, b = table_grey(rows[i]["visual_input"]), table_grey(vids[0])
        n = min(len(a), len(b), 121)
        diffs.append(b[:n] - a[:n])
    n = min(map(len, diffs))
    return np.array([d[:n] for d in diffs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", action="append", required=True, help="label=<prep dir>:<cosmos output dir>")
    ap.add_argument("--num", type=int, default=100, help="segments sampled per batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    summary = {}
    for (label, spec), color in zip((b.split("=", 1) for b in a.batch), [BLUE, ORANGE, INK2, MUTED]):
        prep, out = spec.split(":")
        P = profile(prep, out, a.num, a.seed)
        m, lo, hi = P.mean(0), np.percentile(P, 25, axis=0), np.percentile(P, 75, axis=0)
        x = np.arange(len(m))
        ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)
        ax.plot(x, m, color=color, linewidth=2, label=label)
        ax.text(x[-1] + 1.5, m[-1], f"{label}: {m[-1]:+.0f}", color=INK2, va="center", fontsize=9)
        summary[label] = {"segments": int(P.shape[0]), "mean_by_frame": [round(float(v), 1) for v in m],
                          "frames_darker_than_30": round(float((P < -30).mean()), 3)}
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xlim(0, 150); ax.set_xlabel("frame within the 121-frame segment", color=INK2)
    ax.set_ylabel("output minus source, table grey level", color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2)
    ax.legend(frameon=False, loc="lower left", labelcolor=INK2)
    fig.text(0.02, 0.95, "Cosmos Transfer1 output drifts darker than its source within every segment", fontsize=11.5, color=INK, weight="bold")
    fig.text(0.02, 0.90, f"mean over {a.num} seeded segments per batch, band = interquartile range; frame 0 is the copied conditioning frame", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(a.out, facecolor=SURFACE)
    json.dump(summary, open(a.out.rsplit(".", 1)[0] + ".json", "w"), indent=1)
    print(f"wrote {a.out}; " + "; ".join(f"{k}: end {v['mean_by_frame'][-1]:+.0f}, {v['frames_darker_than_30']:.0%} frames darker by >30" for k, v in summary.items()))


if __name__ == "__main__":
    main()
