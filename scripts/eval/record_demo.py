"""Record demo clips of a robomimic checkpoint: a 512x512 overview camera next to the 200x200 policy view.

Same rollout loop as scripts/eval/shift_eval.py (one env, horizon 800, success checked every step, seed applied once)
so what is shown is the evaluation protocol, but these clips are illustrations: the overview camera is extra and
the rollouts are not the ones counted in results/eval_results.md. Each episode ends with a SUCCESS/FAIL card.
With --eval_type the task must be the shift task; without it the in-distribution scene is used as is.

Episode selection: by default the clip is the first --episodes rollouts in order. With --attempts N (N > episodes)
rollouts continue until --episodes successes are found or N attempts are used, and the clip keeps the successes first
(in attempt order), then failures to fill up to --episodes; the caption on each episode is its attempt number and the
.json next to the clip lists every attempt, so the selection is on record.

  ./isaaclab.sh -p scripts/eval/record_demo.py --task <task> --checkpoint <model.pth> --label "Policy 4: + Cosmos" \
      --condition "two distractors" [--eval_type distractors] --episodes 3 --out clips/p4_distractors.mp4 \
      --headless --enable_cameras
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record demo clips of a robomimic policy.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--eval_type", type=str, default="", help="Shift setting; empty = in-distribution scene.")
parser.add_argument("--label", type=str, default="", help="Policy caption drawn on the clip.")
parser.add_argument("--condition", type=str, default="in-distribution", help="Condition caption.")
parser.add_argument("--horizon", type=int, default=800)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--attempts", type=int, default=0, help="If > 0: roll out up to this many times and keep the successes first (see module docstring).")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--out", type=str, required=True, help="Output .mp4 path; a .json with outcomes is written next to it.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import copy
import json
import os
import random

import cv2
import gymnasium as gym
import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab_tasks.utils import parse_env_cfg

import cosmos_demo.tasks  # noqa: F401  (registers the CosmosDemo-* tasks; must follow the app launch)

SIDE = 512
BAR = 72
FONT = cv2.FONT_HERSHEY_SIMPLEX


def compose(overview, policy_view, label, condition, caption, status=None):
    """overview: HxWx3 uint8 (SIDE x SIDE); policy_view: 200x200x3 uint8. Returns BGR frame for cv2."""
    right = cv2.resize(policy_view, (SIDE, SIDE), interpolation=cv2.INTER_NEAREST)
    frame = np.concatenate([overview, right], axis=1)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    bar = np.zeros((BAR, frame.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, f"{label}  |  {condition}  |  {caption}", (10, 30), FONT, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, "overview camera 512x512 (not seen by the policy)", (10, 58), FONT, 0.5, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(bar, "policy input: table camera 200x200", (SIDE + 10, 58), FONT, 0.5, (170, 170, 170), 1, cv2.LINE_AA)
    frame = np.concatenate([bar, frame], axis=0)
    if status is not None:
        color = (60, 200, 60) if status else (60, 60, 220)
        text = "SUCCESS" if status else "FAIL"
        (tw, th), _ = cv2.getTextSize(text, FONT, 2.0, 4)
        x = (frame.shape[1] - tw) // 2
        y = BAR + SIDE // 2
        cv2.rectangle(frame, (x - 20, y - th - 20), (x + tw + 20, y + 20), (0, 0, 0), -1)
        cv2.putText(frame, text, (x, y), FONT, 2.0, color, 4, cv2.LINE_AA)
    return frame


def rollout(policy, env, success_term, horizon, device, writer, label, condition, caption):
    policy.start_episode()
    obs_dict, _ = env.reset()
    last = None
    for i in range(horizon):
        obs = copy.deepcopy(obs_dict["policy"])
        for ob in obs:
            obs[ob] = torch.squeeze(obs[ob])
        for image_name in env.cfg.image_obs_list:
            if image_name in obs_dict["policy"].keys():
                image = torch.squeeze(obs_dict["policy"][image_name])
                if image_name == "table_cam":
                    policy_view = image.cpu().numpy()[..., :3]
                image = image.permute(2, 0, 1).clone().float() / 255.0
                obs[image_name] = image.clip(0.0, 1.0)
        overview = env.scene["demo_cam"].data.output["rgb"][0].cpu().numpy()[..., :3]
        last = compose(overview, policy_view, label, condition, caption)
        writer.write(last)
        actions = policy(obs)
        actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        if bool(success_term.func(env, **success_term.params)[0]):
            ok, steps = True, i + 1
            break
        elif terminated or truncated:
            ok, steps = False, i + 1
            break
    else:
        ok, steps = False, horizon
    overview = env.scene["demo_cam"].data.output["rgb"][0].cpu().numpy()[..., :3]
    policy_view = torch.squeeze(obs_dict["policy"]["table_cam"]).cpu().numpy()[..., :3]
    card = compose(overview, policy_view, label, condition, caption, status=ok)
    for _ in range(int(1.5 * args_cli.fps)):
        writer.write(card)
    return ok, steps


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    env_cfg.recorders = None
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None
    if args_cli.eval_type:
        env_cfg.eval_mode = True
        env_cfg.eval_type = args_cli.eval_type
    # Overview camera: same pose as the policy's table camera, higher resolution, never observed by the policy.
    env_cfg.scene.demo_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/demo_cam",
        update_period=0.0,
        height=SIDE,
        width=SIDE,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2)),
        offset=CameraCfg.OffsetCfg(pos=(1.0, 0.0, 0.4), rot=(-0.61237, -0.61237, 0.35355, 0.35355), convention="ros"),
    )
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    env.seed(args_cli.seed)
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    base = os.path.splitext(args_cli.out)[0]
    fourcc, size = cv2.VideoWriter_fourcc(*"mp4v"), (2 * SIDE, SIDE + BAR)
    n_attempts = args_cli.attempts if args_cli.attempts > 0 else args_cli.episodes
    attempts = []  # every rollout goes to its own temporary file; the clip is assembled from the kept ones below
    with torch.inference_mode():
        for k in range(1, n_attempts + 1):
            policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device)
            tmp = f"{base}.attempt{k:02d}.mp4"
            writer = cv2.VideoWriter(tmp, fourcc, args_cli.fps, size)
            ok, steps = rollout(policy, env, success_term, args_cli.horizon, device, writer, args_cli.label, args_cli.condition, f"attempt {k}")
            writer.release()
            attempts.append({"attempt": k, "success": ok, "steps": steps, "file": tmp})
            print(f"[DEMO] attempt {k}: {'success' if ok else 'fail'} after {steps} steps")
            if args_cli.attempts > 0 and sum(a["success"] for a in attempts) >= args_cli.episodes:
                break
    if args_cli.attempts > 0:
        kept = [a for a in attempts if a["success"]][: args_cli.episodes]
        kept += [a for a in attempts if not a["success"]][: args_cli.episodes - len(kept)]
    else:
        kept = attempts
    writer = cv2.VideoWriter(args_cli.out, fourcc, args_cli.fps, size)
    for a in kept:
        cap = cv2.VideoCapture(a["file"])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        cap.release()
    writer.release()
    for a in attempts:
        os.remove(a["file"])
    outcomes = [{"episode": i + 1, "attempt": a["attempt"], "success": a["success"], "steps": a["steps"]} for i, a in enumerate(kept)]
    meta = {"task": args_cli.task, "checkpoint": args_cli.checkpoint, "eval_type": args_cli.eval_type or "in-distribution",
            "label": args_cli.label, "condition": args_cli.condition, "seed": args_cli.seed,
            "selection": "successes first" if args_cli.attempts > 0 else "consecutive",
            "attempts": [{k: v for k, v in a.items() if k != "file"} for a in attempts], "episodes": outcomes}
    with open(base + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[DEMO] wrote {args_cli.out}: {sum(o['success'] for o in outcomes)}/{len(outcomes)} successes shown, "
          f"{sum(a['success'] for a in attempts)}/{len(attempts)} attempts succeeded")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
