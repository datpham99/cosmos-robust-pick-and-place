"""Offline sanity check: feed a dataset episode's observations to a robomimic checkpoint and compare
predicted actions to the recorded ones. No simulator. Mirrors play.py's observation preprocessing.
Usage: <isaac python> scripts/diag/check_policy_offline.py <ckpt.pth> <dataset.hdf5> [demo_key=demo_0]
"""
import sys

import h5py
import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils

ckpt, ds = sys.argv[1], sys.argv[2]
demo = sys.argv[3] if len(sys.argv) > 3 else "demo_0"
policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpt, device="cuda", verbose=False)
keys = list(policy.policy.obs_shapes.keys())
print("policy obs keys:", {k: tuple(policy.policy.obs_shapes[k]) for k in keys})
with h5py.File(ds, "r") as f:
    g = f["data"][demo]
    acts = g["actions"][()]
    obs = {k: g["obs"][k][()] for k in keys}
policy.start_episode()
pred = []
for t in range(len(acts)):
    o = {}
    for k in keys:
        v = torch.as_tensor(obs[k][t])
        if v.ndim == 3:  # HWC uint8 -> CHW float in [0,1], as play.py does
            v = v.permute(2, 0, 1).float() / 255.0
        o[k] = v
    pred.append(policy(o))
pred = np.asarray(pred)
err = np.abs(pred - acts)
print(f"steps {len(acts)}  action dims {acts.shape[1]}")
print("recorded |a| mean per dim:", np.round(np.abs(acts).mean(0), 3))
print("pred     |a| mean per dim:", np.round(np.abs(pred).mean(0), 3))
print("mean abs error per dim:   ", np.round(err.mean(0), 3))
print("gripper sign agreement:   ", float(np.mean(np.sign(pred[:, -1]) == np.sign(acts[:, -1]))))
print("xyz direction cosine:     ", float(np.mean(np.sum(pred[:, :3] * acts[:, :3], 1) / (np.linalg.norm(pred[:, :3], axis=1) * np.linalg.norm(acts[:, :3], axis=1) + 1e-8))))
