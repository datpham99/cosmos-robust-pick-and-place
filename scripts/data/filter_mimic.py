"""Keep only episodes whose FINAL recorded frame satisfies the stack success check.

Mimic marks an episode successful at the first frame `cubes_stacked` passes; the stack can topple
during the retreat and the episode is still exported. This applies the same criterion (xy < 4 cm,
height difference within 5 mm of 4.68 cm, upper cube above lower, gripper open) to the last frame
of the recorded `cube_positions` / `gripper_pos` observations and copies the passing episodes.

Usage: python.sh scripts/data/filter_mimic.py <in.hdf5> <out.hdf5> [--max N] [--append]
  --max N    keep at most the first N passing episodes (in demo order)
  --append   add to an existing output file, continuing its demo numbering
"""
import argparse

import h5py
import numpy as np

XY, HTOL, HDIFF, GRIP = 0.04, 0.005, 0.0468, 0.04


def stacked_final(ep):
    cp = ep["obs/cube_positions"][-1].reshape(3, 3)
    ok = True
    for lo, hi in ((0, 1), (1, 2)):
        d = cp[lo] - cp[hi]  # lower cube minus the cube on top of it
        ok &= np.linalg.norm(d[:2]) < XY and (np.linalg.norm(d[2:]) - HDIFF) < HTOL and d[2] < 0
    g = ep["obs/gripper_pos"][-1]
    return bool(ok and abs(g[0] - GRIP) < 1e-3 and abs(abs(g[1]) - GRIP) < 1e-3)


p = argparse.ArgumentParser()
p.add_argument("src"); p.add_argument("out"); p.add_argument("--max", type=int, default=None); p.add_argument("--append", action="store_true")
a = p.parse_args()
src = h5py.File(a.src, "r")
names = sorted(src["data"].keys(), key=lambda k: int(k.split("_")[1]))
keep = [k for k in names if stacked_final(src["data"][k])]
drop = [k for k in names if k not in set(keep)]
if a.max is not None:
    keep = keep[: a.max]
with h5py.File(a.out, "a" if a.append else "w") as out:
    g = out.require_group("data")
    start = len(g)
    if start == 0:
        for k, v in src.attrs.items():  # root attrs: format_version tells Isaac Lab's loader the quaternion order
            out.attrs[k] = v
        for k, v in src["data"].attrs.items():
            g.attrs[k] = v
    total = int(g.attrs["total"]) if start else 0  # the copied source attrs carry the source total
    for i, k in enumerate(keep):
        src["data"].copy(src["data"][k], g, name=f"demo_{start + i}")
        total += int(src["data"][k].attrs["num_samples"])
    g.attrs["total"] = total
    prev = list(g.attrs.get("source_episodes", np.array([], dtype="S")))
    g.attrs["source_episodes"] = np.array(prev + [f"{a.src}:{k}".encode() for k in keep], dtype="S")
    print(f"{a.src}: {len(names)} episodes, {len(names) - len(drop)} pass the final-frame check, kept {len(keep)}; output now {len(g)} episodes, {total} steps")
    print(f"dropped {len(drop)}: {drop}")
