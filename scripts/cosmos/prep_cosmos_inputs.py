"""Prepare Cosmos Transfer1 inputs from the subset HDF5: 121-frame segments, denoised RGB clips, and clean edge-control clips.

Why (see docs/writeup.md, Cosmos Transfer1 on a 32 GB card):
- The tokenizer asserts T % 121 == 0, so a 239-frame episode cannot be run as one clip; each episode is cut into
  121-frame segments (starts 0, 90, ... and a last segment aligned to the end) that are stitched back after generation.
- Transfer1's internal Canny on 200x200 frames loses the cubes and the gripper. The edge control is computed here
  from the RGB (median-filtered only if --median > 0) plus the segmentation-label boundaries, both at the working
  resolution, and passed per row through `control_overrides`.
- Working size 960x704 (Transfer1's 4:3 preset): 1280x704 does not fit in 32 GB on this GPU.

  python scripts/cosmos/prep_cosmos_inputs.py --subset <hdf5> --prompts <batch200.jsonl> --out <dir> [--demos 0 1 ...]
Writes <out>/demo_<i>_s<k>.mp4, demo_<i>_s<k>_edge.mp4 and <out>/batch.jsonl; segments.json records the frame ranges.
"""
import argparse
import json
import os

import cv2
import h5py
import numpy as np

SEG_LEN = 121
STEP = 90  # new segment every 90 frames: Transfer1 anchors colours on each segment's first (original) frame and drifts after about 100 frames
W, H = 960, 704


def segment_starts(T):
    if T <= SEG_LEN:
        return [0]
    starts = list(range(0, T - SEG_LEN, STEP))
    if starts[-1] != T - SEG_LEN:
        starts.append(T - SEG_LEN)
    return starts


def label_map(seg_rgba):
    s = seg_rgba[..., :3].astype(np.int32)
    return s[..., 0] * 65536 + s[..., 1] * 256 + s[..., 2]


def edge_frame(rgb200, seg200, median):
    rgb = cv2.resize(cv2.medianBlur(rgb200, median) if median else rgb200, (W, H), interpolation=cv2.INTER_CUBIC)
    canny = cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 100, 200)
    lab = cv2.resize(label_map(seg200), (W, H), interpolation=cv2.INTER_NEAREST)
    b = np.zeros_like(canny)
    b[:, 1:] |= (lab[:, 1:] != lab[:, :-1]).astype(np.uint8) * 255
    b[1:, :] |= (lab[1:, :] != lab[:-1, :]).astype(np.uint8) * 255
    return rgb, np.maximum(canny, b)


def write_video(path, frames_bgr):
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    for fr in frames_bgr:
        out.write(fr)
    out.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--prompts", required=True, help="batch JSONL whose row i holds demo_i's prompt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--demos", type=int, nargs="*", default=None)
    ap.add_argument("--median", type=int, default=0, help="median filter size, 0 = off; the study used 0 on the patched (clean) renders")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prompts = [json.loads(l)["prompt"] for l in open(a.prompts)]
    f = h5py.File(a.subset, "r")
    demos = a.demos if a.demos is not None else list(range(len(prompts)))
    rows, segs = [], {}
    for i in demos:
        rgb = f[f"data/demo_{i}/obs/table_cam"][:]
        seg = f[f"data/demo_{i}/obs/table_cam_segmentation"][:]
        T = rgb.shape[0]
        segs[f"demo_{i}"] = {"T": T, "starts": segment_starts(T)}
        for k, s in enumerate(segs[f"demo_{i}"]["starts"]):
            vid, edge = [], []
            for t in range(s, min(s + SEG_LEN, T)):
                r, e = edge_frame(rgb[t], seg[t], a.median)
                vid.append(cv2.cvtColor(r, cv2.COLOR_RGB2BGR))
                edge.append(cv2.cvtColor(e, cv2.COLOR_GRAY2BGR))
            base = os.path.join(os.path.abspath(a.out), f"demo_{i}_s{k}")
            write_video(base + ".mp4", vid)
            write_video(base + "_edge.mp4", edge)
            rows.append({"prompt": prompts[i], "visual_input": base + ".mp4",
                         "control_overrides": {"edge": {"input_control": base + "_edge.mp4"}}})
        print(f"demo_{i}: T={T} segments={segs[f'demo_{i}']['starts']}", flush=True)
    with open(os.path.join(a.out, "batch.jsonl"), "w") as fo:
        for r in rows:
            fo.write(json.dumps(r) + "\n")
    json.dump(segs, open(os.path.join(a.out, "segments.json"), "w"), indent=1)
    print(f"{len(rows)} rows -> {a.out}/batch.jsonl")


if __name__ == "__main__":
    main()
