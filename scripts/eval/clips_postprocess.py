"""Everything made from the demo clips (scripts/eval/record_demo.py output), in one place.

  gifs    one small GIF per clip for the README and docs/demos.md: overview panel only, the clip's first episode
          plus its result card, downscaled and subsampled                              -> results/demo_gifs/
  figure  one policy-camera frame per evaluation condition, tiled with labels, from one policy's clips
                                                                                        -> results/figures/shift_conditions.png
  index   docs/demos.md: one section per condition with the four policies' GIFs, the shown episodes and attempt
          counts from the clips' .json files, and the 100-rollout rate from results/eval_results.md
  all     the three above with their default outputs (what scripts/eval/record_demos_all.sh runs when it finishes)

  python.sh scripts/eval/clips_postprocess.py gifs   --clips <dir with p*_*.mp4> [--out results/demo_gifs] [--width 200] [--fps 6] [--max_s 12]
  python.sh scripts/eval/clips_postprocess.py figure --clips <dir> [--policy p4] [--out results/figures/shift_conditions.png] [--frame 40]
  python3   scripts/eval/clips_postprocess.py index  --clips <dir> [<dir> ...] [--results results/eval_results.md] [--out docs/demos.md]
  python.sh scripts/eval/clips_postprocess.py all    --clips <dir> [--policy p4]

`gifs`, `figure` and `all` need cv2 and PIL (Isaac Sim's python.sh); `index` runs on plain Python 3. Output paths
default to this repository, whatever the working directory. For `index`, later clip directories override earlier ones
for the same policy/condition.
"""
import argparse
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CARD_S = 1.5  # seconds of result card after each episode (record_demo.py)

POLICIES = [("p1", "1: 10 demonstrations"), ("p2", "2: + Mimic 500"), ("p3", "3: + domain randomization"), ("p4", "4: + Cosmos Transfer1")]
CONDITIONS = [  # key, short label (figure), title and description (index)
    ("indist", "in-distribution", "Training scene",
     "The training scene: stock table, stock dome light, three cubes, no extra objects."),
    ("camera_offset", "camera offset 5 cm", "Camera offset 5 cm", "The table camera moved by (3, -4, 2) cm, 5.4 cm in all, orientation kept."),
    ("light_color", "light colour", "New light colour",
     "The dome light's colour drawn from the stock randomizer's range at reset."),
    ("light_intensity", "light intensity", "New light intensity",
     "Dome-light intensity drawn from 5000-10000 (held out) instead of 1500-5000 (training half)."),
    ("light_texture", "light texture", "New light texture", "One of the five held-out HDR environment maps on the dome light."),
    ("distractors", "two distractors", "Two distractor objects",
     "A YCB tomato-soup can and mustard bottle dropped beside the cube region."),
    ("all", "all five at once", "All five changes at once", "Every factor above applied together."),
]
FIGURE_ORDER = ["indist", "light_color", "light_texture", "light_intensity", "camera_offset", "distractors", "all"]
# policy 1's reported checkpoint is epoch 160; P1_EPOCH=<n> in the environment selects another evaluation label
INDIST_LABEL = {"p1": f"p1_v2_600_ep{os.environ.get('P1_EPOCH', '160')}", "p2": "p2_v2_600_ep600", "p3": "p3_v2_600_ep600", "p4": "p4_v2_600_ep600"}
SHIFT_LABEL = {"all": "combined"}  # the combined condition's row label differs from its eval_type


def panel_geometry(cap):
    """The clip frame is a caption bar over two square panels (overview left, policy right)."""
    import cv2
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    side = w // 2
    return side, h - side  # SIDE, BAR


def gifs(clips, out, width=200, fps=6, max_s=12.0, panel="overview"):
    import cv2
    from PIL import Image
    os.makedirs(out, exist_ok=True)
    for mp4 in sorted(glob.glob(os.path.join(clips, "p*_*.mp4"))):
        meta_path = os.path.splitext(mp4)[0] + ".json"
        if not os.path.exists(meta_path):
            continue  # clip still being written
        meta = json.load(open(meta_path))
        cap = cv2.VideoCapture(mp4)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        SIDE, BAR = panel_geometry(cap)
        ep = meta["episodes"][0]
        card = int(CARD_S * src_fps)
        n_ep = ep["steps"] + card                        # episode frames + result card
        n_keep = min(n_ep, int(max_s * src_fps) + card)
        stride = max(1, round(src_fps / fps))
        frames = []
        for i in range(n_ep):
            ok, f = cap.read()
            if not ok:
                break
            if n_keep - card <= i < n_ep - card:
                continue  # skip the middle of over-long failed episodes, keep the card
            if (i % stride) and i < n_ep - card:
                continue
            if panel == "overview":
                p = f[BAR:, :SIDE]
            elif panel == "policy":
                p = f[BAR:, SIDE:]
            else:
                p = f[BAR:]
            p = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
            w = width * (2 if panel == "both" else 1)
            p = cv2.resize(p, (w, int(p.shape[0] * w / p.shape[1])), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(p).quantize(colors=128, method=Image.Quantize.MEDIANCUT))
        cap.release()
        if not frames:
            continue
        gif = os.path.join(out, os.path.basename(mp4).replace(".mp4", ".gif"))
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
        print(f"{gif}: {len(frames)} frames, {os.path.getsize(gif) / 1e6:.2f} MB, first episode {'success' if ep['success'] else 'fail'}")


def figure(clips, policy, out, frame=40):
    import cv2
    import numpy as np
    labels = {key: label for key, label, _, _ in CONDITIONS}
    tiles = []
    for key in FIGURE_ORDER:
        mp4 = os.path.join(clips, f"{policy}_{key}.mp4")
        cap = cv2.VideoCapture(mp4)
        if not cap.isOpened():
            print(f"[figure] missing {mp4}, tile skipped")
            continue
        SIDE, BAR = panel_geometry(cap)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, f = cap.read()
        assert ok, key
        tile = cv2.resize(f[BAR:, SIDE:], (256, 256), interpolation=cv2.INTER_AREA)  # policy panel (200x200 upscaled)
        strip = np.full((28, 256, 3), 20, dtype=np.uint8)
        cv2.putText(strip, labels[key], (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.concatenate([tile, strip], 0))
    if not tiles:
        return
    per_row = 4
    while len(tiles) % per_row:
        tiles.append(np.full_like(tiles[0], 20))  # blank tile pads the last row
    sheet = np.concatenate([np.concatenate(tiles[i:i + per_row], 1) for i in range(0, len(tiles), per_row)], 0)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, sheet)
    print(out, sheet.shape)


def index(clip_dirs, results, out):
    rates = {}
    for line in open(results):
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 8 or not re.match(r"p[1-4]_v2_600", cells[2]):
            continue
        rates[cells[2]] = cells[6]

    def rate(p, c):
        if c == "indist":
            return rates.get(INDIST_LABEL[p], "not evaluated")
        return rates.get(f"{p}_v2_600_shift_{SHIFT_LABEL.get(c, c)}", "not evaluated")

    def outcomes(p, c):
        meta = None
        for d in clip_dirs:
            f = os.path.join(d, f"{p}_{c}.json")
            if os.path.exists(f):
                meta = json.load(open(f))
        if meta is None:
            return "not recorded"
        marks = " ".join("S" if e["success"] else "F" for e in meta["episodes"])
        att = meta.get("attempts")
        if att and meta.get("selection") == "successes first":
            return f"{marks}; {sum(a['success'] for a in att)}/{len(att)} attempts succeeded"
        return f"{sum(e['success'] for e in meta['episodes'])}/{len(meta['episodes'])} ({marks})"

    lines = [
        "# Clips: every policy under every condition",
        "",
        "Three episodes per evaluation, environment seed 0, recorded with `scripts/eval/record_demo.py`. The overview "
        "camera sits at the policy camera's pose at 512x512 and is never seen by the policy. A result card follows each "
        "episode. The episodes are chosen successes first from up to ten rollouts, the same rule for every policy. A "
        "GIF opens on a success wherever the policy managed one. The caption on each episode is its "
        "attempt number. Each GIF shows the clip's first episode, overview panel, capped at 12 s. The full clips with all "
        "three episodes and the 200x200 policy view are among the hosted files (`tutorial.md`, section 6). \"Clip\" lists "
        "the three shown episodes, S or F in clip order, and how many attempts succeeded. \"100 rollouts\" is the matching "
        "row of `results/eval_results.md`. That row is the number to trust. The clips only illustrate it.",
        "",
        "The study reports policy 1 only in the training scene and under the combined shift, 0 of 100 in both. Its "
        "single-shift clips are included for completeness.",
        "",
    ]
    for key, _, title, desc in CONDITIONS:
        lines += [f"## {title}", "", desc, ""]
        lines.append("| " + " | ".join(name for _, name in POLICIES) + " |")
        lines.append("|" + "---|" * len(POLICIES))
        lines.append("| " + " | ".join(f"![{p} {key}](../results/demo_gifs/{p}_{key}.gif)" for p, _ in POLICIES) + " |")
        lines.append("| " + " | ".join(f"clip {outcomes(p, key)}" for p, _ in POLICIES) + " |")
        lines.append("| " + " | ".join(f"100 rollouts: {rate(p, key)}" for p, _ in POLICIES) + " |")
        lines.append("")
    open(out, "w").write("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gifs"); g.add_argument("--clips", required=True); g.add_argument("--out", default=f"{ROOT}/results/demo_gifs")
    g.add_argument("--width", type=int, default=200); g.add_argument("--fps", type=int, default=6)
    g.add_argument("--max_s", type=float, default=12.0, help="Cap on the episode length kept, seconds of source video.")
    g.add_argument("--panel", choices=["overview", "policy", "both"], default="overview")
    f = sub.add_parser("figure"); f.add_argument("--clips", required=True); f.add_argument("--policy", default="p4")
    f.add_argument("--out", default=f"{ROOT}/results/figures/shift_conditions.png"); f.add_argument("--frame", type=int, default=40)
    i = sub.add_parser("index"); i.add_argument("--clips", nargs="+", required=True)
    i.add_argument("--results", default=f"{ROOT}/results/eval_results.md"); i.add_argument("--out", default=f"{ROOT}/docs/demos.md")
    a = sub.add_parser("all"); a.add_argument("--clips", required=True); a.add_argument("--policy", default="p4")
    args = ap.parse_args()
    if args.cmd == "gifs":
        gifs(args.clips, args.out, args.width, args.fps, args.max_s, args.panel)
    elif args.cmd == "figure":
        figure(args.clips, args.policy, args.out, args.frame)
    elif args.cmd == "index":
        index(args.clips, args.results, args.out)
    else:
        gifs(args.clips, f"{ROOT}/results/demo_gifs")
        figure(args.clips, args.policy, f"{ROOT}/results/figures/shift_conditions.png")
        index([args.clips], f"{ROOT}/results/eval_results.md", f"{ROOT}/docs/demos.md")
