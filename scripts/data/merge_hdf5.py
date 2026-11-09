"""Concatenate HDF5 demo files into one, keeping the root attributes (format_version, see tutorial T11) and recomputing
`total`. Isaac Lab's merge_hdf5_datasets.py copies only `env_args`, so a merged file could not be replayed.

  python.sh scripts/data/merge_hdf5.py <out.hdf5> <in1.hdf5> <in2.hdf5> ...
"""
import os
import sys

import h5py
import numpy as np


def main():
    out, ins = sys.argv[1], sys.argv[2:]
    n, total, sources = 0, 0, []
    with h5py.File(out, "w") as g:
        data = g.create_group("data")
        for j, path in enumerate(ins):
            with h5py.File(path, "r") as f:
                if j == 0:
                    for k, v in f.attrs.items():
                        g.attrs[k] = v
                    for k, v in f["data"].attrs.items():
                        if k not in ("source_episodes", "total"):
                            data.attrs[k] = v
                for k in sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1])):
                    f.copy(f"data/{k}", data, name=f"demo_{n}")
                    total += int(f[f"data/{k}"].attrs["num_samples"])
                    sources.append(f"{os.path.basename(path)}:{k}")
                    n += 1
        data.attrs["total"] = total
        data.attrs["source_episodes"] = np.array(sources, dtype="S")
    print(f"{n} episodes, {total} steps -> {out}")


if __name__ == "__main__":
    main()
