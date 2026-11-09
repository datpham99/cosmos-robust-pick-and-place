"""Contact sheet of first table-camera frames from an HDF5 dataset, for the visual checks in docs/tutorial.md section 3.

Picks N episodes evenly spaced through the file and tiles their first `obs/table_cam` frame, ten per row, with the
episode name drawn on each tile. The study's sheets are in results/figures/data_samples/.

  <isaac python> scripts/diag/first_frames.py <file.hdf5> <out.png> [--n 20] [--frame 0]
"""
import argparse

import cv2
import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("hdf5")
ap.add_argument("out")
ap.add_argument("--n", type=int, default=20, help="episodes to show, evenly spaced through the file")
ap.add_argument("--frame", type=int, default=0, help="frame index to show from each episode")
a = ap.parse_args()

with h5py.File(a.hdf5, "r") as f:
    names = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
    picks = [names[i] for i in np.linspace(0, len(names) - 1, min(a.n, len(names))).astype(int)]
    tiles = []
    for k in picks:
        cam = f[f"data/{k}/obs/table_cam"]
        img = np.array(cam[min(a.frame, cam.shape[0] - 1)])
        cv2.putText(img, k, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
rows = [np.concatenate(tiles[i:i + 10], 1) for i in range(0, len(tiles), 10)]
w = max(r.shape[1] for r in rows)
rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
cv2.imwrite(a.out, cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
print(f"wrote {a.out}: {len(tiles)} episodes of {len(names)}, frame {a.frame}")
