"""Draw the Cosmos subset: N episodes of a Mimic HDF5 chosen with random.seed(seed), copied to a new file as
demo_0..demo_{N-1} (root format_version, data attrs and per-demo attrs preserved) and the draw order saved as JSON.

  python.sh scripts/cosmos/make_cosmos_subset.py <in.hdf5> <out.hdf5> <order.json> [--num 200] [--seed 0]
"""
import argparse
import json
import os
import random

import h5py
import numpy as np


def draw(ids, num, seed):
    random.seed(seed)
    return random.sample(ids, num)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("order_json")
    ap.add_argument("--num", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    with h5py.File(a.src, "r") as f:
        ids = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        chosen = draw(ids, a.num, a.seed)
        with h5py.File(a.out, "w") as g:
            for k, v in f.attrs.items():
                g.attrs[k] = v
            data = g.create_group("data")
            total = 0
            for i, k in enumerate(chosen):
                f.copy(f"data/{k}", data, name=f"demo_{i}")
                total += int(f[f"data/{k}"].attrs["num_samples"])
            for k, v in f["data"].attrs.items():
                if k not in ("source_episodes", "total"):
                    data.attrs[k] = v
            data.attrs["source_episodes"] = np.array([f"{os.path.basename(a.src)}:{k}" for k in chosen], dtype="S")
            data.attrs["total"] = total
    json.dump({"source": a.src, "seed": a.seed, "order": chosen}, open(a.order_json, "w"), indent=1)
    print(f"{len(chosen)} episodes, {total} steps -> {a.out}; order in {a.order_json}")


if __name__ == "__main__":
    main()
