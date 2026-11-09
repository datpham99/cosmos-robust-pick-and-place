"""Stitch Cosmos Transfer1 segment outputs (video_<row>/output.mp4) back into one clip per episode.

Rows of <prep>/batch.jsonl are demo_<i>_s<k> segments written by scripts/cosmos/prep_cosmos_inputs.py; segments.json holds each
episode's frame count T and segment start frames. Output frame t comes from the first segment that reaches it, except
that each later segment's first frame (Transfer1 copies the conditioning frame there un-augmented) is skipped, which
the 1-frame overlap of consecutive segments allows. Writes <out>/demo_<i>_table_cam.mp4 (T frames, segment resolution)
and prints episodes whose segments are missing or short.

  python scripts/cosmos/stitch_cosmos_outputs.py --prep <prep_dir> --cosmos_out <dir with video_*/> --out <dir>
"""
import argparse
import json
import os
import re

import cv2


def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--cosmos_out", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    segs = json.load(open(os.path.join(a.prep, "segments.json")))
    rows = [json.loads(l) for l in open(os.path.join(a.prep, "batch.jsonl"))]
    by_demo = {}
    for r, row in enumerate(rows):
        m = re.search(r"(demo_\d+)_s(\d+)\.mp4$", row["visual_input"])
        by_demo.setdefault(m.group(1), []).append((int(m.group(2)), os.path.join(a.cosmos_out, f"video_{r}", "output.mp4")))
    n_ok = 0
    for demo, parts in sorted(by_demo.items(), key=lambda kv: int(kv[0].split("_")[1])):
        T, starts = segs[demo]["T"], segs[demo]["starts"]
        parts.sort()
        if len(parts) != len(starts) or any(not os.path.exists(p) for _, p in parts):
            print(f"{demo}: missing segment output, skipped")
            continue
        frames, bad = [], False
        for (k, path), s in zip(parts, starts):
            seg = read_frames(path)
            if len(seg) < min(121, T - s):
                print(f"{demo}: segment {k} has {len(seg)} frames, expected {min(121, T - s)}, skipped")
                bad = True
                break
            first = len(frames) - s  # index into this segment of the first frame not yet covered
            if k > 0:
                first = max(first, 1)  # skip the copied conditioning frame
            frames.extend(seg[first:min(121, T - s)])
        if bad or len(frames) != T:
            if not bad:
                print(f"{demo}: stitched {len(frames)} frames, expected {T}, skipped")
            continue
        h, w = frames[0].shape[:2]
        out = cv2.VideoWriter(os.path.join(a.out, f"{demo}_table_cam.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        for fr in frames:
            out.write(fr)
        out.release()
        n_ok += 1
    print(f"stitched {n_ok}/{len(by_demo)} episodes -> {a.out}")


if __name__ == "__main__":
    main()
