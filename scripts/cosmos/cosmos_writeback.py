"""Write stitched Cosmos clips back into a copy of the subset HDF5 (actions and every other key untouched).

Why not Isaac Lab's mp4_to_hdf5.py: it appends the clips as extra demos next to the originals with no
frame-count check, and the edge-distilled Transfer1 swaps a cube's colour for another cube's as soon as two cubes touch
(most episodes fail the 40-degree hue gate, 10-40 % of each cube's frames; edge control carries no
colour information). Dropping those leaves 9 episodes, so instead the cube colour is restored from the simulator: inside
each cube's segmentation mask the hue and saturation come from the original frame and the value (shading) from the
Cosmos frame. Everything outside the cubes is Cosmos output. The pre- and post-correction hue drift of every episode is
recorded in the report so the correction rate is reported, not hidden. Episodes failing the silhouette gate
(alignment_report.json, `min` < threshold) are dropped, as the protocol requires.

  python.sh scripts/cosmos/cosmos_writeback.py --subset <subset.hdf5> --videos_dir <stitched> --report <alignment_report.json> \
      --out <out.hdf5> --report_out <json> [--threshold 0.6] [--hue_tol 40] [--no_restore]
"""
import argparse
import json
import os
import sys

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cosmos_alignment_check import hue_deg  # noqa: E402


def read_clip(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(cv2.resize(fr, (200, 200), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    return np.stack(frames)


def cube_labels(orig200, seg_rgba):
    """Integer label map and the saturated-object labels beyond background/table/robot (cube_hue_ok's selection,
    on a 1-D integer map instead of np.unique over RGBA rows, which was 90 s per episode)."""
    s = seg_rgba[..., :3].astype(np.int32)
    lab = s[..., 0] * 65536 + s[..., 1] * 256 + s[..., 2]
    vals, counts = np.unique(lab, return_counts=True)
    cubes = []
    for v in vals[np.argsort(-counts)[3:]]:
        m = lab == v
        if hue_deg(orig200[m].mean(0))[1] >= 60:
            cubes.append(m)
    return cubes


def drift(orig200, frame200, masks):
    worst = 0.0
    for m in masks:
        if m.sum() < 30:
            continue
        h0 = hue_deg(orig200[m].mean(0))[0]
        h1 = hue_deg(frame200[m].mean(0))[0]
        d = abs(h0 - h1)
        worst = max(worst, min(d, 360 - d))
    return worst


def restore(orig200, gen200, masks):
    if not masks:
        return gen200
    m = np.logical_or.reduce(masks)
    ho = cv2.cvtColor(orig200, cv2.COLOR_RGB2HSV)
    hg = cv2.cvtColor(gen200, cv2.COLOR_RGB2HSV)
    hg[m, 0] = ho[m, 0]
    hg[m, 1] = ho[m, 1]
    return cv2.cvtColor(hg, cv2.COLOR_HSV2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--videos_dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report_out", required=True)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--hue_tol", type=float, default=40.0)
    ap.add_argument("--no_restore", action="store_true")
    a = ap.parse_args()
    align = json.load(open(a.report))
    out_rep, kept, total = {}, [], 0
    with h5py.File(a.subset, "r") as f, h5py.File(a.out, "w") as g:
        for k, v in f.attrs.items():
            g.attrs[k] = v
        data = g.create_group("data")
        ids = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
        for src in ids:
            rec = align.get(src)
            if rec is None or rec["min"] is None:
                out_rep[src] = {"dropped": "no stitched clip or no alignment entry"}
                print(f"{src}: dropped, not in the alignment report (missing or short Cosmos output)", flush=True)
                continue
            if rec["min"] < a.threshold:
                out_rep[src] = {"dropped": "silhouette", "edge_min": rec["min"]}
                print(f"{src}: dropped, silhouette min {rec['min']:.2f}", flush=True)
                continue
            orig = f[f"data/{src}/obs/table_cam"][:]
            seg = f[f"data/{src}/obs/table_cam_segmentation"][:]
            T = orig.shape[0]
            gen = read_clip(os.path.join(a.videos_dir, f"{src}_table_cam.mp4"))
            assert gen.shape[0] >= T, f"{src}: clip has {gen.shape[0]} frames, episode {T}"
            gen = gen[:T]
            before, after, n_bad = 0.0, 0.0, 0
            fixed = np.empty_like(orig)
            for t in range(T):
                masks = cube_labels(orig[t], seg[t])
                d0 = drift(orig[t], gen[t], masks)
                fixed[t] = gen[t] if a.no_restore else restore(orig[t], gen[t], masks)
                d1 = drift(orig[t], fixed[t], masks)
                before, after = max(before, d0), max(after, d1)
                n_bad += d0 > a.hue_tol
            dst = f"demo_{len(kept)}"
            f.copy(f"data/{src}", data, name=dst)
            del data[f"{dst}/obs/table_cam"]
            d = f[f"data/{src}/obs/table_cam"]
            data.create_dataset(f"{dst}/obs/table_cam", data=fixed, dtype=d.dtype, chunks=d.chunks,
                                compression=d.compression, compression_opts=d.compression_opts)
            total += T
            kept.append(src)
            out_rep[src] = {"written_as": dst, "edge_min": rec["min"], "hue_drift_before": before,
                            "hue_drift_after": after, "frames_over_tol_before": n_bad, "T": T}
            print(f"{src} -> {dst}: edge min {rec['min']:.2f}, hue drift {before:.0f} -> {after:.0f} deg, "
                  f"{n_bad}/{T} frames corrected", flush=True)
        for k, v in f["data"].attrs.items():
            if k not in ("source_episodes", "total"):
                data.attrs[k] = v
        data.attrs["source_episodes"] = np.array([f"{os.path.basename(a.subset)}:{k}" for k in kept], dtype="S")
        data.attrs["total"] = total
        data.attrs["cosmos_hue_restored"] = not a.no_restore
    json.dump({"subset": a.subset, "videos_dir": a.videos_dir, "threshold": a.threshold, "hue_tol": a.hue_tol,
               "restore": not a.no_restore, "episodes": out_rep}, open(a.report_out, "w"), indent=1)
    n_fix = sum(1 for r in out_rep.values() if r.get("hue_drift_before", 0) > a.hue_tol)
    n_after = sum(1 for r in out_rep.values() if r.get("hue_drift_after", 0) > a.hue_tol)
    print(f"{len(kept)}/{len(ids)} episodes written to {a.out} ({total} steps); {n_fix} needed hue restoration, "
          f"{n_after} still over {a.hue_tol} deg after it", flush=True)


if __name__ == "__main__":
    main()
