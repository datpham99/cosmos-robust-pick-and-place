"""Render test: dump table_cam frames of a task under a chosen antialiasing mode / RTX render mode.

Checks the render fix for the per-pixel speckle of the stock Cosmos visuomotor config on Isaac Sim 6.0.1
(docs/writeup.md, Notes before you start). Runs K steps with a slow sinusoidal end-effector motion (so temporal AA artifacts show), saves
frames at the listed steps as one contact sheet, and prints a speckle score (mean |Laplacian| of the grayscale frame;
clean renders score low).

  cd ~/IsaacLab && ./isaaclab.sh -p <repo>/scripts/setup/render_test.py --task <task> --headless --enable_cameras \
      --label stock [--aa DLAA] [--rendermode RaytracedLighting] [--steps 40] [--out results/figures/render_test]
"""
import argparse
import importlib
import os
import sys

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--aa", default=None, help="antialiasing_mode override: Off, FXAA, DLSS, TAA, DLAA")
parser.add_argument("--rendermode", default=None, help="/rtx/rendermode override via carb_settings, e.g. RaytracedLighting")
parser.add_argument("--rendermode_post", default=None, help="/rtx/rendermode set after env creation (after the AA call)")
parser.add_argument("--steps", type=int, default=40)
parser.add_argument("--dump", default="0,5,20,40")
parser.add_argument("--out", default="results/figures/render_test")
parser.add_argument("--external_callback", default=None)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args



def load_callback(name):
    """Accept a module ("cosmos_demo.tasks") or a module.function path ("cosmos_demo.tasks.register"), as Isaac Lab does."""
    try:
        importlib.import_module(name)
    except ModuleNotFoundError:
        mod, _, fn = name.rpartition(".")
        getattr(importlib.import_module(mod), fn)()


def main():
    import cv2

    torch.manual_seed(0)
    if args_cli.external_callback:
        load_callback(args_cli.external_callback)
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    env_cfg.scene.num_envs = 1
    if args_cli.aa:
        env_cfg.sim.render.antialiasing_mode = args_cli.aa
    if args_cli.rendermode:
        cs = dict(env_cfg.sim.render.carb_settings or {})
        cs["/rtx/rendermode"] = args_cli.rendermode
        env_cfg.sim.render.carb_settings = cs
    print(f"[RT] {args_cli.label}: antialiasing_mode={env_cfg.sim.render.antialiasing_mode} "
          f"carb_settings={env_cfg.sim.render.carb_settings}", flush=True)
    dump = {int(s) for s in args_cli.dump.split(",")}
    os.makedirs(args_cli.out, exist_ok=True)
    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        if args_cli.rendermode_post:
            import carb
            carb.settings.get_settings().set("/rtx/rendermode", args_cli.rendermode_post)
            print(f"[RT] /rtx/rendermode set to {args_cli.rendermode_post} after env creation", flush=True)
        obs, _ = env.reset()
        tiles, scores = [], []

        def grab(step):
            fr = obs["policy"]["table_cam"][0].detach().cpu().numpy()
            if fr.dtype != np.uint8:
                fr = (fr * 255).clip(0, 255).astype(np.uint8)
            fr = fr[..., :3]
            lap = cv2.Laplacian(cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY), cv2.CV_32F)
            scores.append((step, float(np.abs(lap).mean())))
            tiles.append(cv2.resize(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), (400, 400), interpolation=cv2.INTER_NEAREST))

        grab(0)
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        with torch.inference_mode():
            for t in range(1, args_cli.steps + 1):
                actions[:, 0] = 0.4 * np.sin(t / 6.0)
                actions[:, 2] = -0.3 * np.cos(t / 6.0)
                obs, _, _, _, _ = env.step(actions)
                if t in dump:
                    grab(t)
        sheet = np.concatenate(tiles, 1)
        path = os.path.join(args_cli.out, f"{args_cli.label}.png")
        cv2.imwrite(path, sheet)
        print(f"[RT] {args_cli.label}: speckle scores (mean |Laplacian|) " +
              ", ".join(f"step {s}: {v:.2f}" for s, v in scores) + f" -> {path}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
