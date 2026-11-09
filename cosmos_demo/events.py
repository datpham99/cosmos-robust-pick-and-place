"""Shift-set event terms added by this project: the camera offset and the two distractor objects.

Both are gated on ``eval_type`` the same way Isaac Lab's stock lighting randomization is, so a single
``eval_type`` value selects one factor and ``all`` turns every factor on at once.
"""

from __future__ import annotations

import random

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

def offset_table_cam(
    env: ManagerBasedEnv, env_ids: torch.Tensor, sensor_cfg: SceneEntityCfg, offset_pos: tuple[float, float, float]
):
    """Shift-set factor ``camera_offset``: move the table camera by a fixed few centimetres (orientation kept).
    Rewrites the pose from the cfg every reset so the camera is back at its default when inactive."""
    cam = env.scene[sensor_cfg.name]
    active = env.cfg.eval_mode and env.cfg.eval_type in ["camera_offset", "all"]
    pos = env.scene.env_origins[env_ids] + torch.tensor(cam.cfg.offset.pos, device=env.device)
    if active:
        pos = pos + torch.tensor(offset_pos, device=env.device)
    rot = torch.tensor(cam.cfg.offset.rot, device=env.device).unsqueeze(0).repeat(len(env_ids), 1)
    cam.set_world_poses(pos, rot, env_ids=env_ids.tolist(), convention=cam.cfg.offset.convention)


def place_distractors(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_names: list[str],
    x_range: tuple[float, float],
    y_abs_range: tuple[float, float],
    z_drop: float,
    park_pos: tuple[float, float, float],
):
    """Shift-set factor ``distractors``: drop the listed rigid objects onto the table beside the cube region,
    one on each side (sides swapped at random), with a random yaw. Inactive: parked out of view below the table."""
    active = env.cfg.eval_mode and env.cfg.eval_type in ["distractors", "all"]
    for i in env_ids.tolist():
        flip = random.random() < 0.5
        for j, name in enumerate(asset_names):
            obj = env.scene[name]
            if active:
                side = 1.0 if (j == 0) != flip else -1.0
                pos = (random.uniform(*x_range), side * random.uniform(*y_abs_range), z_drop)
                yaw = random.uniform(-3.14159, 3.14159)
                quat = (torch.cos(torch.tensor(yaw / 2)).item(), 0.0, 0.0, torch.sin(torch.tensor(yaw / 2)).item())
            else:
                pos, quat = park_pos, (1.0, 0.0, 0.0, 0.0)
            state = torch.zeros((1, 13), device=env.device)
            state[0, :3] = env.scene.env_origins[i] + torch.tensor(pos, device=env.device)
            state[0, 3:7] = torch.tensor(quat, device=env.device)
            ids = torch.tensor([i], device=env.device)
            obj.write_root_pose_to_sim(state[:, :7], ids)
            obj.write_root_velocity_to_sim(state[:, 7:], ids)
