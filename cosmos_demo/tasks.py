"""Task registrations for this project. Loaded into Isaac Lab scripts via ``--external_callback``.

Example:
  ./isaaclab.sh -p scripts/tools/record_demos.py --task CosmosDemo-Stack-Cube-Franka-IK-Rel-v0 \
      --teleop_device pulse_keyboard --external_callback cosmos_demo.tasks.register ...
"""

import gymnasium as gym

from isaaclab.devices import DevicesCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.manager_based.manipulation.stack.config.franka import agents
from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_ik_rel_env_cfg import FrankaCubeStackEnvCfg
from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_ik_rel_visuomotor_cosmos_env_cfg import (
    FrankaCubeStackVisuomotorCosmosEnvCfg,
)
from isaaclab_mimic.envs.franka_stack_ik_rel_visuomotor_cosmos_mimic_env_cfg import (
    FrankaCubeStackIKRelVisuomotorCosmosMimicEnvCfg,
)

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from cosmos_demo.events import offset_table_cam, place_distractors
from cosmos_demo.teleop import PulseKeyboardCfg
from cosmos_demo.textures import HELDOUT, TRAIN, apply_split


@configclass
class TeleopStackEnvCfg(FrankaCubeStackEnvCfg):
    """Stock IK-relative stack task plus the remote-desktop-safe keyboard device."""

    def __post_init__(self):
        super().__post_init__()
        # same sensitivities record_demos.py uses for its built-in keyboard
        self.teleop_devices = DevicesCfg(
            devices={"pulse_keyboard": PulseKeyboardCfg(pos_sensitivity=0.2, rot_sensitivity=0.5)}
        )


gym.register(
    id="CosmosDemo-Stack-Cube-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}:TeleopStackEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
    disable_env_checker=True,
)


@configclass
class DRTrainCosmosMimicEnvCfg(FrankaCubeStackIKRelVisuomotorCosmosMimicEnvCfg):
    """Cosmos-Mimic stack env with appearance randomization forced ON at every reset, drawing only
    from the TRAIN half of the lighting split. Used to re-render the Mimic dataset for policy 3."""

    eval_mode = True
    eval_type = "all"

    def __post_init__(self):
        super().__post_init__()
        apply_split(self.events, TRAIN)


@configclass
class ShiftEvalCosmosEnvCfg(FrankaCubeStackVisuomotorCosmosEnvCfg):
    """Cosmos training env whose randomization lists hold only the HELD-OUT half of the split.
    scripts/eval/shift_eval.py sets eval_mode and selects eval_type."""

    def __post_init__(self):
        super().__post_init__()
        apply_split(self.events, HELDOUT)
        # Two further shift factors of the held-out shift set (docs/writeup.md, Setup); none of these values or assets
        # appear in any training data. Both are gated on eval_type like the stock ones.
        self.events.offset_table_cam = EventTerm(
            func=offset_table_cam,
            mode="reset",
            params={"sensor_cfg": SceneEntityCfg("table_cam"), "offset_pos": (0.03, -0.04, 0.02)},
        )
        props = RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                                       max_angular_velocity=1000.0, max_linear_velocity=1000.0, max_depenetration_velocity=5.0,
                                       disable_gravity=False)
        self.scene.distractor_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Distractor_1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, -3.0], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd", rigid_props=props),
        )
        self.scene.distractor_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Distractor_2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, -3.0], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd", rigid_props=props),
        )
        self.events.place_distractors = EventTerm(
            func=place_distractors,
            mode="reset",
            params={"asset_names": ["distractor_1", "distractor_2"], "x_range": (0.42, 0.58),
                    "y_abs_range": (0.19, 0.23), "z_drop": 0.12, "park_pos": (0.5, 0.0, -3.0)},
        )


gym.register(
    id="CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-DR-Mimic-v0",
    entry_point="isaaclab_mimic.envs.franka_stack_ik_rel_mimic_env:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}:DRTrainCosmosMimicEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_image_cosmos.json",
    },
    disable_env_checker=True,
)

gym.register(
    id="CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Shift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}:ShiftEvalCosmosEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_image_cosmos.json",
    },
    disable_env_checker=True,
)


def register() -> list:
    """External callback entry point. Importing this module performs the registration."""
    return []


@configclass
class TopUpCosmosMimicEnvCfg(FrankaCubeStackIKRelVisuomotorCosmosMimicEnvCfg):
    """Stock Cosmos-Mimic generation env with a different datagen seed, used to generate replacement
    episodes for the ones dropped by the final-frame success filter (scripts/data/filter_mimic.py)."""

    def __post_init__(self):
        super().__post_init__()
        self.datagen_config.seed = 2


gym.register(
    id="CosmosDemo-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Mimic-Seed2-v0",
    entry_point="isaaclab_mimic.envs.franka_stack_ik_rel_mimic_env:FrankaCubeStackIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}:TopUpCosmosMimicEnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_image_cosmos.json",
    },
    disable_env_checker=True,
)
