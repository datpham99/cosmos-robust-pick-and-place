"""Evaluate a robomimic checkpoint under one visual-shift setting.

Same rollout protocol as Isaac Lab's ``scripts/imitation_learning/robomimic/play.py`` (one env, horizon
800, success checked every step, seed applied once before the rollouts) so the shift column is directly
comparable with the in-distribution column. Differences: the project task module is registered after the
app launches, ``eval_mode`` is forced on and ``--eval_type`` picks the setting (the stock randomization
events are no-ops otherwise), and the first frame of the first ``--save_frames`` rollouts is written to a
contact sheet so the applied shift can be inspected.

  ./isaaclab.sh -p scripts/eval/shift_eval.py --task CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0 \
      --eval_type all --checkpoint <model.pth> --num_rollouts 100 --seed 0 --headless --enable_cameras
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a robomimic policy under a visual-shift setting.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--eval_type", type=str, default="all")
parser.add_argument("--horizon", type=int, default=800)
parser.add_argument("--num_rollouts", type=int, default=1)
parser.add_argument("--seed", type=int, default=101)
parser.add_argument("--save_frames", type=int, default=8, help="Write the first frame of this many rollouts.")
parser.add_argument("--frames_out", type=str, default=None, help="Contact-sheet PNG path.")
parser.add_argument("--frame_steps", type=str, default="0", help="Comma list of step indices to capture per saved rollout.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import copy
import random

import gymnasium as gym
import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import torch

from isaaclab_tasks.utils import parse_env_cfg

import cosmos_demo.tasks  # noqa: F401  (registers the CosmosDemo-* tasks; must follow the app launch)

FRAME_STEPS = {int(x) for x in args_cli.frame_steps.split(",")}


def rollout(policy, env, success_term, horizon, device, first_frame):
    policy.start_episode()
    obs_dict, _ = env.reset()
    for i in range(horizon):
        obs = copy.deepcopy(obs_dict["policy"])
        for ob in obs:
            obs[ob] = torch.squeeze(obs[ob])
        if hasattr(env.cfg, "image_obs_list"):
            for image_name in env.cfg.image_obs_list:
                if image_name in obs_dict["policy"].keys():
                    image = torch.squeeze(obs_dict["policy"][image_name])
                    if i in FRAME_STEPS and image_name == "table_cam" and first_frame is not None:
                        first_frame.append(image.cpu().numpy())
                    image = image.permute(2, 0, 1).clone().float()
                    image = image / 255.0
                    image = image.clip(0.0, 1.0)
                    obs[image_name] = image
        actions = policy(obs)
        actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        if bool(success_term.func(env, **success_term.params)[0]):
            return True
        elif terminated or truncated:
            return False
    return False


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    env_cfg.recorders = None
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None
    env_cfg.eval_mode = True
    env_cfg.eval_type = args_cli.eval_type
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    print(f"[SHIFT] eval_mode={env.cfg.eval_mode} eval_type={env.cfg.eval_type}")

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    env.seed(args_cli.seed)
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    frames = []
    with torch.inference_mode():
        results = []
        for trial in range(args_cli.num_rollouts):
            print(f"[INFO] Starting trial {trial}")
            policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device)
            ok = rollout(policy, env, success_term, args_cli.horizon, device, frames if trial < args_cli.save_frames else None)
            results.append(ok)
            print(f"[INFO] Trial {trial}: {ok}\n")

    if frames and args_cli.frames_out:
        import PIL.Image as I
        I.fromarray(np.concatenate(frames, 1)).save(args_cli.frames_out)
        print(f"[SHIFT] wrote {len(frames)} first frames to {args_cli.frames_out}")

    print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
    print(f"Success rate: {results.count(True) / len(results)}")
    print(f"Trial Results: {results}\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
