"""Replay recorded episodes open-loop in a 1-env Mimic env and print how far the simulated state drifts
from the recorded one, step by step. Used to see whether a replay failure starts at step 0 (systematic)
or grows gradually (physics divergence).
  ./isaaclab.sh -p scripts/diag/diag_replay.py --headless --enable_cameras --input_file X.hdf5 --episodes demo_0,demo_1
"""
import argparse
import sys

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Mimic-v0")
parser.add_argument("--input_file", required=True)
parser.add_argument("--episodes", default="demo_0")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def main():
    launch_cfg, _ = resolve_task_config("Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-v0", "")
    with launch_simulation(launch_cfg, args_cli):
        import isaaclab_mimic.envs  # noqa: F401
        from isaaclab.utils.datasets import HDF5DatasetFileHandler

        env_cfg, _ = resolve_task_config(args_cli.task, "")
        env_cfg.scene.num_envs = 1
        success_term = env_cfg.terminations.success
        env_cfg.terminations = None
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        fh = HDF5DatasetFileHandler(); fh.open(args_cli.input_file)
        with torch.inference_mode():
            for name in args_cli.episodes.split(","):
                ep = fh.load_episode(name, env.device)
                actions = ep.data["actions"]
                rec_eef = ep.data["obs"]["eef_pos"]
                rec_cube = {k: ep.data["states"]["rigid_object"][k]["root_pose"] for k in ("cube_1", "cube_2", "cube_3")}
                env.sim.reset(); env.reset_to(ep.data["initial_state"], None, is_relative=True)
                checkpoints = {0, 1, 5, 20, 50, 100, 150, len(actions) - 1}
                print(f"[DIAG] {name}: {len(actions)} steps; env origin {env.scene.env_origins[0].tolist()}")
                jp_sim = env.scene["robot"].data.joint_pos[0].cpu().numpy(); jp_rec = ep.data["states"]["articulation"]["robot"]["joint_position"][0].cpu().numpy()
                jp_init = ep.data["initial_state"]["articulation"]["robot"]["joint_position"].reshape(-1).cpu().numpy()
                print(f"[DIAG]   after reset: sim joints {np.round(jp_sim,3).tolist()}\n[DIAG]   rec states[0] {np.round(jp_rec,3).tolist()}\n[DIAG]   initial_state {np.round(jp_init,3).tolist()}")
                rb = env.scene["robot"].data
                print(f"[DIAG]   sim robot root pos {np.round(rb.root_pos_w[0].cpu().numpy(),3).tolist()} quat {np.round(rb.root_quat_w[0].cpu().numpy(),3).tolist()}  rec init root_pose {np.round(ep.data['initial_state']['articulation']['robot']['root_pose'].reshape(-1).cpu().numpy(),3).tolist()}")
                print(f"[DIAG]   sim ee_frame pos {np.round((env.scene['ee_frame'].data.target_pos_w[0,0]-env.scene.env_origins[0]).cpu().numpy(),3).tolist()}  rec eef_pos[0] {np.round(rec_eef[0].cpu().numpy(),3).tolist()}")
                for i, a in enumerate(actions):
                    obs, *_ = env.step(a.reshape(1, -1))
                    if i in checkpoints:
                        eef = obs["policy"]["eef_pos"][0]
                        d_eef = float(torch.norm(eef - rec_eef[i]))
                        if i == 0:
                            print(f"[DIAG]   step 0 sim obs eef {np.round(eef.cpu().numpy(),3).tolist()} rec {np.round(rec_eef[0].cpu().numpy(),3).tolist()}")
                        d_cubes = []
                        for k in ("cube_1", "cube_2", "cube_3"):
                            p = env.scene[k].data.root_pos_w[0] - env.scene.env_origins[0]
                            d_cubes.append(float(torch.norm(p - rec_cube[k][i][:3])))
                        print(f"[DIAG]   step {i:3d}: |eef - rec| = {d_eef:.4f}  |cube - rec| = {[round(d, 4) for d in d_cubes]}  gripper act {float(a[-1]):+.0f}")
                ok = bool(success_term.func(env, **success_term.params)[0])
                final = [np.round((env.scene[k].data.root_pos_w[0] - env.scene.env_origins[0]).cpu().numpy(), 3).tolist() for k in ("cube_1", "cube_2", "cube_3")]
                print(f"[DIAG]   success={ok} final cubes {final} recorded final {[np.round(rec_cube[k][-1][:3].cpu().numpy(), 3).tolist() for k in ('cube_1','cube_2','cube_3')]}")
        env.close()


if __name__ == "__main__":
    main()
