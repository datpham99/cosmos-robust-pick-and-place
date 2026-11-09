"""Alignment gate for Cosmos outputs (the protocol in docs/writeup.md, Setup): overlay augmented frames on the original segmentation mask.

For each augmented clip, N frames are drawn at random (seeded). Two checks per frame: edge alignment (below) and cube
colour: for every segmentation label other than the three largest (background, table, robot), the hue of the mean
colour inside that label's mask in the augmented frame must stay within --hue_tol degrees of the hue in the original
frame, so a cube that Cosmos recoloured fails the clip (the policy keys on cube colour). The original episode's table_cam_segmentation gives
the foreground silhouette (robot + cubes = every pixel that is neither background nor the table colour). The Cosmos
frame is resized to 200x200 and its Canny edges are compared with the silhouette boundary: the score is the fraction
of silhouette-boundary pixels that have an augmented-frame edge within 2 px. A contact sheet per clip
(augmented frame with the silhouette boundary drawn in green) is written for the manual pass. The clip passes if every
sampled frame scores >= --threshold; the gate in the protocol is a visual match in all sampled frames, so the sheets are
the record and the score is the pre-filter.

  python scripts/cosmos/cosmos_alignment_check.py --subset <subset.hdf5> --videos_dir <cosmos_out> --out_dir <dir> [--frames 20]
"""
import argparse
import glob
import json
import os
import random

import cv2
import h5py
import numpy as np


def silhouette(seg_rgba):
    flat = seg_rgba.reshape(-1, seg_rgba.shape[-1])
    vals, counts = np.unique(flat, axis=0, return_counts=True)
    order = np.argsort(-counts)
    background = vals[order[0]]  # most frequent: black (unlabelled) or the table, whichever is larger
    table = vals[order[1]]
    m = ~((seg_rgba == background).all(-1) | (seg_rgba == table).all(-1))
    return m.astype(np.uint8)


def hue_deg(rgb_mean):
    h = cv2.cvtColor(np.uint8([[rgb_mean]]), cv2.COLOR_RGB2HSV)[0, 0]
    return float(h[0]) * 2.0, float(h[1])


def cube_hue_ok(orig200, frame200, seg_rgba, tol):
    flat = seg_rgba.reshape(-1, seg_rgba.shape[-1])
    vals, counts = np.unique(flat, axis=0, return_counts=True)
    order = np.argsort(-counts)
    worst = 0.0
    for v in vals[order[3:]]:  # skip background, table, robot
        m = (seg_rgba == v).all(-1)
        if m.sum() < 30:
            continue
        h0, s0 = hue_deg(orig200[m].mean(0))
        if s0 < 60:  # not a saturated object
            continue
        h1, _ = hue_deg(frame200[m].mean(0))
        d = abs(h0 - h1)
        worst = max(worst, min(d, 360 - d))
    return worst <= tol, worst


def boundary(mask):
    k = np.ones((3, 3), np.uint8)
    return (cv2.dilate(mask, k) - cv2.erode(mask, k)) > 0


def score(frame200, mask):
    b = boundary(mask)
    if b.sum() == 0:
        return 1.0
    edges = cv2.Canny(cv2.cvtColor(frame200, cv2.COLOR_RGB2GRAY), 50, 150) > 0
    near = cv2.dilate(edges.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return float((b & near).sum() / b.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--videos_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hue_tol", type=float, default=40.0, help="max hue drift (degrees) of any cube")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    random.seed(a.seed)
    f = h5py.File(a.subset, "r")
    report = {}
    for vid in sorted(glob.glob(os.path.join(a.videos_dir, "**", "*.mp4"), recursive=True)):
        name = os.path.basename(vid).replace("_table_cam", "").replace(".mp4", "")
        demo = name if name.startswith("demo_") else None
        if demo is None or demo not in f["data"]:
            continue
        seg = f["data"][demo]["obs/table_cam_segmentation"]
        orig = f["data"][demo]["obs/table_cam"]
        cap = cv2.VideoCapture(vid)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = sorted(random.sample(range(min(n, seg.shape[0])), min(a.frames, n)))
        tiles, scores, hues = [], [], []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if not ok:
                continue
            fr = cv2.cvtColor(cv2.resize(fr, (200, 200), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            m = silhouette(seg[i])
            s = score(fr, m)
            scores.append(s)
            hues.append(cube_hue_ok(orig[i], fr, seg[i], a.hue_tol)[1])
            tile = fr.copy()
            tile[boundary(m)] = (0, 255, 0)
            tiles.append(tile)
        cap.release()
        if tiles:
            rows = [np.concatenate(tiles[j:j + 10], 1) for j in range(0, len(tiles), 10)]
            w = max(r.shape[1] for r in rows)
            rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
            cv2.imwrite(os.path.join(a.out_dir, f"{demo}_overlay.png"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
        report[demo] = {"frames": idx, "scores": [round(s, 3) for s in scores],
                        "min": round(min(scores), 3) if scores else None,
                        "max_hue_drift": round(max(hues), 1) if hues else None,
                        "pass": bool(scores) and min(scores) >= a.threshold and max(hues) <= a.hue_tol}
        print(f"{demo}: edge min {report[demo]['min']} mean {np.mean(scores):.3f}; cube hue drift max {report[demo]['max_hue_drift']} "
              f"-> {'PASS' if report[demo]['pass'] else 'FAIL'}")
    json.dump(report, open(os.path.join(a.out_dir, "alignment_report.json"), "w"), indent=1)
    n_pass = sum(r["pass"] for r in report.values())
    n_sil = sum(r["min"] is not None and r["min"] >= a.threshold for r in report.values())
    print(f"{n_sil}/{len(report)} clips pass the silhouette gate (edge threshold {a.threshold}). "
          f"{n_pass}/{len(report)} also pass the cube-hue gate ({a.hue_tol} deg). "
          f"The write-back drops clips on the silhouette gate only and restores cube hue itself.")


if __name__ == "__main__":
    main()
