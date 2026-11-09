"""Headline chart: success rate of the four policies, in-distribution and under all five shifts at once, with the
95% Wilson interval, read from results/eval_results.md (the rows eval_policy.sh appends).

  python3 scripts/eval/make_chart.py --out results/headline.png \
      --indist p1=p1_v2_600_ep160 p2=p2_v2_600_ep600 p3=p3_v2_600_ep600 p4=p4_v2_600_ep600 \
      --shift  p1=p1_v2_600_shift_combined p2=p2_v2_600_shift_combined p3=p3_v2_600_shift_combined p4=p4_v2_600_shift_combined \
      [--title "..."] [--subtitle "..."]
A policy missing from either list leaves its slot empty and is noted under the chart. Also writes <out>.svg.
"""
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

POLICIES = [("p1", "1: 10 demonstrations"), ("p2", "2: + Mimic 500"),
            ("p3", "3: + domain randomization"), ("p4", "4: + Cosmos Transfer1")]
# palette: slot 1 blue, slot 2 orange; ink and chrome
BLUE, ORANGE, INK, INK2, MUTED, GRID, AXIS, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
ROW = re.compile(r"^\| (\S+) \| (\S+) \| (\S+) \| (\S+) \| seed (\d+) \| (\d+)/(\d+) \| ([0-9.]+) \| \[([0-9.]+), ([0-9.]+)\]")


def load(path):
    out = {}
    for line in open(path):
        m = ROW.match(line)
        if m:
            out[m.group(2)] = dict(k=int(m.group(6)), n=int(m.group(7)), p=float(m.group(8)), lo=float(m.group(9)), hi=float(m.group(10)))
    return out


def rounded_bar(ax, x, h, w, color):
    if h <= 0:
        return
    r = min(0.012, h / 2)
    ax.add_patch(FancyBboxPatch((x - w / 2, 0), w, h - r, boxstyle=f"round,pad=0,rounding_size=0", lw=0, fc=color))
    ax.add_patch(FancyBboxPatch((x - w / 2, h - 2 * r), w, 2 * r, boxstyle=f"round,pad=0,rounding_size={r}", lw=0, fc=color))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/eval_results.md")
    ap.add_argument("--out", required=True)
    ap.add_argument("--indist", nargs="*", default=[])
    ap.add_argument("--shift", nargs="*", default=[])
    ap.add_argument("--title", default="Stack-cube success rate, in-distribution vs. all five held-out shifts at once")
    ap.add_argument("--subtitle", default="BC-RNN, table camera only. 100 sequential rollouts from seed 0, the same sequence for every policy. Whiskers: 95% Wilson interval")
    ap.add_argument("--policies", nargs="*", default=None,
                    help="override the policy list: id=label, with | as a line break (default: the four-policy study)")
    a = ap.parse_args()
    global POLICIES
    if a.policies:
        POLICIES = [(s.split("=", 1)[0], s.split("=", 1)[1].replace("|", "\n")) for s in a.policies]
    rows = load(a.results)
    sel = {"indist": dict(s.split("=") for s in a.indist), "shift": dict(s.split("=") for s in a.shift)}
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 10})
    fig, ax = plt.subplots(figsize=(max(9, 2.25 * len(POLICIES)), 5), dpi=160)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    w, gap, missing = 0.09, 0.06, []  # <= 24 px at 160 dpi
    for i, (pid, name) in enumerate(POLICIES):
        for j, (cond, color, label) in enumerate([("indist", BLUE, "in-distribution"), ("shift", ORANGE, "all five shifts at once")]):
            x = i + (j - 0.5) * (w + gap)
            lab = sel[cond].get(pid)
            r = rows.get(lab) if lab else None
            if r is None:
                missing.append(f"policy {pid[1]} {label}: {'not evaluated' if not lab else 'no row ' + lab}")
                ax.text(x, 0.012, "n/a", ha="center", va="bottom", color=MUTED, fontsize=9)
                continue
            rounded_bar(ax, x, r["p"], w, color)
            ax.plot([x, x], [r["lo"], r["hi"]], color=INK, lw=1.2, solid_capstyle="butt", zorder=3)
            ax.plot([x - 0.04, x + 0.04], [r["lo"], r["lo"]], color=INK, lw=1.2, zorder=3)
            ax.plot([x - 0.04, x + 0.04], [r["hi"], r["hi"]], color=INK, lw=1.2, zorder=3)
            ax.text(x, r["hi"] + 0.015, f"{r['k']}", ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_xlim(-0.6, len(POLICIES) - 0.4); ax.set_ylim(0, 1.0)
    ax.set_xticks(range(len(POLICIES))); ax.set_xticklabels([n for _, n in POLICIES], color=INK2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0", "25", "50", "75", "100 %"], color=MUTED)
    ax.yaxis.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS); ax.tick_params(length=0)
    ax.set_ylabel("success rate", color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=BLUE, lw=0), plt.Rectangle((0, 0), 1, 1, fc=ORANGE, lw=0)]
    ax.legend(handles, ["in-distribution", "all five shifts at once"], frameon=False, loc="upper left", labelcolor=INK2)
    fig.text(0.09, 0.95, a.title, fontsize=12, color=INK, weight="bold")
    fig.text(0.09, 0.905, a.subtitle, fontsize=8.5, color=INK2)
    if missing:
        fig.text(0.09, 0.01, "Not shown: " + "; ".join(missing), fontsize=8, color=MUTED)
    fig.subplots_adjust(top=0.84, bottom=0.2 if missing else 0.16, left=0.09, right=0.98)
    fig.savefig(a.out); fig.savefig(a.out.rsplit(".", 1)[0] + ".svg")
    print(f"wrote {a.out}; missing: {missing}")


if __name__ == "__main__":
    main()
