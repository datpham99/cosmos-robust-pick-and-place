"""Merge domain-randomized re-render passes into one dataset aligned with the original Mimic set.

Each re-render pass (annotate_demos.py --auto) drops episodes whose replay did not reach the success
condition, so one pass yields fewer episodes than the source. This script matches every re-rendered
episode to its source episode by byte-identical actions, keeps the first occurrence of each source
episode, and writes them in source order as demo_0..demo_{N-1}.

Usage: python.sh scripts/data/merge_dr.py [--fill] <source.hdf5> <output.hdf5> <pass1.hdf5> [pass2.hdf5 ...]
With --fill, source episodes missing from every pass are copied from the source unchanged (original
rendering), so the output holds exactly the source trajectories; the output records which ones.
"""
import hashlib
import sys

import h5py
import numpy as np


def ahash(grp):
    return hashlib.md5(np.ascontiguousarray(grp["actions"][()]).tobytes()).hexdigest()


args = sys.argv[1:]
fill = args[0] == "--fill"
src_path, out_path, *passes = args[1:] if fill else args
src = h5py.File(src_path, "r")
src_names = sorted(src["data"].keys(), key=lambda k: int(k.split("_")[1]))
src_hash = {ahash(src["data"][k]): k for k in src_names}
assert len(src_hash) == len(src_names), "source has duplicate action sequences"

found = {}  # source name -> (file, demo name)
unmatched = 0
files = [h5py.File(p, "r") for p in passes]
for f in files:
    for k in f["data"]:
        s = src_hash.get(ahash(f["data"][k]))
        if s is None:
            unmatched += 1  # e.g. a re-rendered episode whose source was later filtered out
            continue
        found.setdefault(s, (f, k))

missing = [k for k in src_names if k not in found]
if fill:
    for k in missing:
        found[k] = (src, k)
with h5py.File(out_path, "w") as out:
    g = out.create_group("data")
    for a, v in files[0].attrs.items():  # root attrs: format_version tells Isaac Lab's loader the quaternion order
        out.attrs[a] = v
    for a, v in files[0]["data"].attrs.items():
        g.attrs[a] = v
    total = 0
    for i, s in enumerate(k for k in src_names if k in found):
        f, k = found[s]
        f["data"].copy(f["data"][k], g, name=f"demo_{i}")
        total += int(f["data"][k].attrs["num_samples"])
    g.attrs["total"] = total
    g.attrs["source_demo_names"] = np.array([k for k in src_names if k in found], dtype="S")
    g.attrs["filled_from_source"] = np.array(missing if fill else [], dtype="S")
print(f"skipped {unmatched} re-rendered episodes with no source match")
print(f"merged {len(found)}/{len(src_names)} episodes, {total} steps -> {out_path}")
print(f"{'filled from source' if fill else 'missing'} {len(missing)}: {missing}")
