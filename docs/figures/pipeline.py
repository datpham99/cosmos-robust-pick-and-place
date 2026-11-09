"""Draw docs/figures/pipeline.png, the pipeline overview in docs/tutorial.md section 0.2.

  python3 docs/figures/pipeline.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, MUTED, EDGE, FILL, HEAD, ACCENT = "#0b0b0b", "#52514e", "#898781", "#f4f3ee", "#e1e0d9", "#2a78d6"
fig, ax = plt.subplots(figsize=(17, 7.2), dpi=150)
ax.set_xlim(0, 17.2); ax.set_ylim(0, 7.2); ax.axis("off")


def box(x0, y0, w, h, lines, bold_first=False, fill=FILL, edge=EDGE, size=10.5):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.12", fc=fill, ec=edge, lw=1.2))
    cx, cy = x0 + w / 2, y0 + h / 2
    n = len(lines); step = 0.3
    for i, t in enumerate(lines):
        ax.text(cx, cy + (n - 1) * step / 2 - i * step, t, ha="center", va="center", fontsize=size,
                color=INK if i == 0 else MUTED, fontweight="bold" if (bold_first and i == 0) else "normal")


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3, shrinkA=0, shrinkB=0))


def header(x0, w, text):
    ax.add_patch(FancyBboxPatch((x0, 6.45), w, 0.5, boxstyle="round,pad=0,rounding_size=0.1", fc=HEAD, ec="none"))
    ax.text(x0 + w / 2, 6.7, text, ha="center", va="center", fontsize=11.5, color=INK, fontweight="bold")


rows = {1: 5.05, 2: 3.6, 3: 2.15, 4: 0.7}       # bottom y of each policy row's boxes
H = 0.95
header(0.3, 2.4, "Source"); header(3.2, 3.9, "3  Data"); header(7.6, 3.7, "4  Training")
header(11.8, 2.4, "5  Evaluation"); header(14.7, 2.4, "Outputs")

# source
box(0.3, 3.3, 2.4, 1.55, ["NVIDIA's 10 human", "teleop demonstrations", "nvidia_teleop_10.hdf5", "3 MB"], bold_first=True, size=9.5)

# data boxes
data = {
    1: ["3.1  Replay with cameras on", "10 episodes with frames"],
    2: ["3.2  GR00T-Mimic generation", "560 generated, 500 kept", "fixed scene"],
    3: ["3.3  Re-render the 500 with", "randomized lighting", "(training half of the split)"],
    4: ["3.4  Cosmos Transfer1 on 200", "of the 500, 199 pass the gate", "500 + 199 episodes; GPU alone"],
}
for r, lines in data.items():
    box(3.2, rows[r], 3.9, H, lines, bold_first=True)

# arrows source -> rows 1 and 2; row 2 -> rows 3 and 4 via a bus on the left
arrow(2.7, 4.3, 3.2, rows[1] + H / 2)
arrow(2.7, 3.9, 3.2, rows[2] + H / 2)
bus_x = 2.95
ax.plot([3.2, bus_x], [rows[2] + 0.15, rows[2] + 0.15], color=MUTED, lw=1.3)
ax.plot([bus_x, bus_x], [rows[2] + 0.15, rows[4] + H / 2], color=MUTED, lw=1.3)
arrow(bus_x, rows[3] + H / 2, 3.2, rows[3] + H / 2)
arrow(bus_x, rows[4] + H / 2, 3.2, rows[4] + H / 2)
ax.text(2.88, (rows[3] + rows[4] + H) / 2 + 0.1, "same 500\nepisodes", ha="right", va="center", fontsize=8.5, color=MUTED)

# training boxes
names = {1: "Policy 1: 10 demonstrations", 2: "Policy 2: + Mimic 500", 3: "Policy 3: + domain randomization", 4: "Policy 4: + Cosmos Transfer1"}
for r in rows:
    box(7.6, rows[r], 3.7, H, [names[r], "BC-RNN, ResNet-18, 600 epochs", "table camera only"], bold_first=True, size=10)
    arrow(7.1, rows[r] + H / 2, 7.6, rows[r] + H / 2)
    arrow(11.3, rows[r] + H / 2, 11.8, rows[r] + H / 2)

# evaluation box spanning all rows
box(11.8, rows[4], 2.4, rows[1] + H - rows[4],
    ["100 rollouts per evaluation", "seed 0, same rollout sequence", "", "7 conditions:", "training scene", "camera offset 5 cm", "light colour", "light intensity",
     "light texture", "two distractors", "all five at once"], bold_first=True, size=10)

# outputs
outs = [(5.2, ["results/eval_results.md", "one row per evaluation"]), (3.45, ["results/headline.png", "the chart"]),
        (1.7, ["docs/demos.md", "a clip per evaluation"])]
for y, lines in outs:
    box(14.7, y, 2.4, 0.85, lines, bold_first=True, size=9.5)
    arrow(14.2, y + 0.425, 14.7, y + 0.425)

ax.text(0.3, 0.15, "Section numbers refer to docs/tutorial.md. Isaac Lab's own scripts do the simulation, generation and training; "
        "scripts/ holds the glue for every arrow.", fontsize=9, color=MUTED, va="center")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(out)
