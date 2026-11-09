"""Train / held-out split of Isaac Lab's stock stacking-task lighting randomization.

The stock visuomotor stack env randomizes the dome light (HDR texture, intensity, colour) only when
``eval_mode`` is on. This project needs two disjoint halves: one the domain-randomization control
(policy 3) may train on, one every policy is evaluated on. Split rule: even positions in the stock HDR
list train, odd positions are held out. Light intensity is split by range; light colour is a continuous
perturbation and is drawn from the same distribution in both halves (stated as a limitation).

The stock env also ships material randomizers for the table and the robot. Both are switched off here:
at runtime a material (re)bind makes the RTX renderer draw those prims at stale transforms in Isaac Sim
6.0.1 with Fabric on (docs/writeup.md, Notes before you start), so the randomization is lighting-only.
"""

from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_ik_rel_visuomotor_env_cfg import EventCfg

_stock = EventCfg()

LIGHT_TEXTURES = _stock.randomize_light.params["textures"]

TRAIN = {
    "light_textures": LIGHT_TEXTURES[0::2],
    "intensity_range": (1500.0, 5000.0),
}
HELDOUT = {
    "light_textures": LIGHT_TEXTURES[1::2],
    "intensity_range": (5000.0, 10000.0),
}


def apply_split(events, split: dict) -> None:
    """Point an env cfg's lighting randomization at one half of the split and switch the stock material randomizers off.

    ``events`` is the cfg's ``EventCfg`` instance; the terms are edited in place.
    """
    events.randomize_light.params["textures"] = list(split["light_textures"])
    events.randomize_light.params["intensity_range"] = tuple(split["intensity_range"])
    events.randomize_table_visual_material = None  # off: runtime material binds corrupt the render
    events.randomize_robot_arm_visual_texture = None  # same reason
