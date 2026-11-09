"""Bounded smoke test for an Isaac Lab task: run N zero-action steps, report observation shapes,
GPU memory peak and steps/s, then exit. Isaac Lab's own zero_agent.py loops forever when headless.

Run with Isaac Sim's bundled Python, from ~/IsaacLab, with conda deactivated:
  ./isaaclab.sh -p <repo>/scripts/diag/smoke_env.py \
      --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 --enable_cameras --headless --num_envs 4 --steps 200
"""

import argparse
import importlib
import sys
import time

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli

parser = argparse.ArgumentParser(description="Bounded smoke test.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--external_callback", default=None, help="Module to import for extra task registrations.")
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
    torch.manual_seed(0)
    if args_cli.external_callback:
        load_callback(args_cli.external_callback)
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env = gym.make(args_cli.task, cfg=env_cfg)
        obs, _ = env.reset()
        print("[SMOKE] observation terms:")
        for group, terms in obs.items():
            if isinstance(terms, dict):
                for k, v in terms.items():
                    print(f"[SMOKE]   {group}/{k}: {tuple(v.shape)} {v.dtype}")
            else:
                print(f"[SMOKE]   {group}: {tuple(terms.shape)} {terms.dtype}")
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.inference_mode():
            for _ in range(args_cli.steps):
                env.step(actions)
        dt = time.time() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"[SMOKE] {args_cli.steps} steps x {args_cli.num_envs} envs in {dt:.1f}s "
              f"= {args_cli.steps / dt:.1f} env-steps/s, torch peak {peak_gb:.2f} GB")
        env.close()
    print("[SMOKE] done")


if __name__ == "__main__":
    main()
