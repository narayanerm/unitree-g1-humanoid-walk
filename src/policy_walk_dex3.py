"""G1-with-Dex3 flat-velocity locomotion (walk) variant.

Drop-in replacement for policy_walk.register / g1_flat_env_cfg that swaps the
29-DOF G1 base body for the G1 + Dex3-1 hand (43 DOF total: 29 base + 14 hand).
Same MDP (rewards, terminations, commands, sensors) as policy_walk; only the
robot entity, the joint tuple and the action-scale dict change.

Prereq: Dex3 XML + meshes installed into mjlab's asset dir:
    bash scripts/install_dex3_assets.sh

Then in a notebook:
    from src.policy_walk_dex3 import register
    TASK_ID = register()
"""
from __future__ import annotations

import shutil
import mujoco
from pathlib import Path

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
    BuiltinPositionActuatorCfg,
    EntityArticulationInfoCfg,
    KNEES_BENT_KEYFRAME,
)
from mjlab.entity import EntityCfg

InitialStateCfg = EntityCfg.InitialStateCfg

import src.policy_walk as _pw


TASK_ID = "Mjlab-Velocity-Flat-Unitree-G1-Dex3"

import mjlab as _mjlab
MJLAB_XMLS = Path(_mjlab.__file__).parent / "asset_zoo" / "robots" / "unitree_g1" / "xmls"
DEX3_XML = MJLAB_XMLS / "g1_with_hand.xml"


def install_dex3_assets(verbose: bool = True) -> Path:
    """Copy the Dex3 XML + STL meshes from the repo into mjlab's asset dir.

    Idempotent — overwrites existing files so a stale copy gets refreshed.
    Mirrors scripts/install_dex3_assets.sh so the notebook can call it without a shell.
    Args: verbose: when True, prints a short install summary.
    Returns: path to the installed g1_with_hand.xml.
    """
    for _p in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (_p / "LICENSE").exists():
            repo_root = _p
            break
    else:
        raise FileNotFoundError("Could not locate repo root (LICENSE missing).")

    src_dir = repo_root / "assets" / "dex3"
    if not src_dir.is_dir():
        raise FileNotFoundError(f"{src_dir} not found. Dex3 assets missing from repo.")
    if not MJLAB_XMLS.is_dir():
        raise FileNotFoundError(f"{MJLAB_XMLS} not found. Is mjlab installed?")

    shutil.copy2(src_dir / "g1_with_hand.xml", MJLAB_XMLS / "g1_with_hand.xml")
    meshes_dst = MJLAB_XMLS / "assets"
    meshes_dst.mkdir(exist_ok=True)
    n = 0
    for stl in (src_dir / "meshes").glob("*.STL"):
        shutil.copy2(stl, meshes_dst / stl.name)
        n += 1
    if verbose:
        print(f"[dex3] installed g1_with_hand.xml + {n} meshes into {MJLAB_XMLS}")
    return DEX3_XML


def _dex3_spec() -> mujoco.MjSpec:
    """Load the G1 + Dex3 MJCF from the installed asset path.
    Args: none.
    Returns: MjSpec parsed from g1_with_hand.xml; auto-installs the assets if missing.
    """
    if not DEX3_XML.exists():
        install_dex3_assets(verbose=False)
    return mujoco.MjSpec.from_file(str(DEX3_XML))


DEX3_HAND_JOINTS = (
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)

# Dex3 finger joints are small and low-inertia: modest stiffness with a high damping ratio keeps them from oscillating under contact. Tune via DR or a sweep once the walk policy itself is stable.
# High stiffness + damping intentionally lock the fingers near the hand-closed home pose
# for walk_dex3: there is no useful finger reward signal here, so the 14 extra action dims
# would otherwise pollute action_rate_l2 and drag the policy into the "stand and fall"
# local optimum (see outputs/logs/g1_walk_dex3/2026-11-07_21-50-13: reward plateaued at
# +2.4 vs base walk +78). carry_dex3 drops these back to 5.0 / 0.5 since it needs mobile
# fingers to grasp the box.
DEX3_HAND_STIFFNESS = 50.0
DEX3_HAND_DAMPING   = 5.0
DEX3_HAND_EFFORT    = 2.5
DEX3_HAND_ARMATURE  = 1.0e-4

# Hand-closed resting pose (fingers flexed toward palm). Z-axis finger joints flip sign
# left-right because their ranges are negatives of each other (per assets/dex3/g1_with_hand.xml).
DEX3_HOME_JOINT_POS = {
    r".*_hand_thumb_0_joint":  0.0,
    r"left_hand_thumb_1_joint":   0.3,
    r"right_hand_thumb_1_joint": -0.3,
    r"left_hand_thumb_2_joint":   0.8,
    r"right_hand_thumb_2_joint": -0.8,
    r"left_hand_index_0_joint":  -1.0,
    r"right_hand_index_0_joint":  1.0,
    r"left_hand_index_1_joint":  -1.2,
    r"right_hand_index_1_joint":  1.2,
    r"left_hand_middle_0_joint": -1.0,
    r"right_hand_middle_0_joint": 1.0,
    r"left_hand_middle_1_joint": -1.2,
    r"right_hand_middle_1_joint": 1.2,
}

G1_DEX3_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hand_(thumb|index|middle)_[0-2]_joint",),
    stiffness=DEX3_HAND_STIFFNESS,
    damping=DEX3_HAND_DAMPING,
    effort_limit=DEX3_HAND_EFFORT,
    armature=DEX3_HAND_ARMATURE,
)


def _dex3_init_state() -> InitialStateCfg:
    """Build the init_state for walk_dex3 by extending KNEES_BENT_KEYFRAME with the hand-closed home pose.
    Args: none.
    Returns: InitialStateCfg inheriting pos/rot/lin_vel/ang_vel/joint_vel from KNEES_BENT_KEYFRAME, with joint_pos merged from the base regex dict and DEX3_HOME_JOINT_POS.
    """
    merged_joint_pos = dict(KNEES_BENT_KEYFRAME.joint_pos)
    merged_joint_pos.update(DEX3_HOME_JOINT_POS)
    return InitialStateCfg(
        pos=KNEES_BENT_KEYFRAME.pos,
        rot=KNEES_BENT_KEYFRAME.rot,
        lin_vel=KNEES_BENT_KEYFRAME.lin_vel,
        ang_vel=KNEES_BENT_KEYFRAME.ang_vel,
        joint_pos=merged_joint_pos,
        joint_vel=KNEES_BENT_KEYFRAME.joint_vel,
    )


def _dex3_robot_cfg() -> EntityCfg:
    """Build the EntityCfg for the G1 + Dex3 body.
    Args: none.
    Returns: EntityCfg combining the base G1 articulation with the Dex3 hand actuator block; init_state locks fingers in the hand-closed home pose and the high stiffness of G1_DEX3_ACTUATOR keeps them there.
    """
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ARTICULATION, FULL_COLLISION
    articulation = EntityArticulationInfoCfg(
        actuators=tuple(G1_ARTICULATION.actuators) + (G1_DEX3_ACTUATOR,),
        soft_joint_pos_limit_factor=G1_ARTICULATION.soft_joint_pos_limit_factor,
    )
    return EntityCfg(
        init_state=_dex3_init_state(),
        collisions=(FULL_COLLISION,),
        spec_fn=_dex3_spec,
        articulation=articulation,
    )


def _dex3_joint_tuple() -> tuple[str, ...]:
    """Concatenate base G1_JOINTS with the Dex3 hand joints.
    Args: none.
    Returns: tuple of joint names preserving the base MuJoCo order followed by DEX3_HAND_JOINTS.
    """
    base = tuple(_pw.G1_JOINTS)
    return base + DEX3_HAND_JOINTS


def _dex3_action_scale() -> dict[str, float]:
    """Extend G1_ACTION_SCALE with per-joint scales for the Dex3 hand.
    Args: none.
    Returns: dict mapping each joint to its action scale; hand joints get 0.25 * effort / stiffness to match the base convention.
    """
    scale = dict(_pw.G1_ACTION_SCALE)
    for j in DEX3_HAND_JOINTS:
        scale[j] = 0.25 * DEX3_HAND_EFFORT / DEX3_HAND_STIFFNESS
    return scale


def _extend_pose_stds_for_hands(cfg) -> None:
    """Append hand-joint entries to the variable_posture reward std dicts.
    Args: cfg: env cfg whose rewards.pose has std_walking / std_running dicts expanded per joint via regex.
    Returns: none; without a .*_hand_.* key the std tensor is shorter than the joint slice and a dim mismatch fires at step time.
    """
    pose_term = cfg.rewards.get("pose")
    if pose_term is None:
        return
    # std_standing uses a catch-all ".*" so only std_walking and std_running need an explicit .*_hand_.* entry; the 0.08 rad band (~4.6 deg) is tight because low-mass fingers read as jittery at the default 0.25 rad band.
    for key in ("std_walking", "std_running"):
        d = pose_term.params.get(key)
        if isinstance(d, dict):
            d[r".*_hand_.*"] = d.get(r".*_hand_.*", 0.08)


def _add_hand_still_reward(cfg) -> None:
    """Add an L1 joint-deviation penalty scoped to the 14 Dex3 hand joints.
    Args: cfg: env cfg whose rewards dict receives a new "hand_still" term with weight -0.05.
    Returns: none; the L1 gradient complements the saturating Gaussian pose reward and suppresses residual finger twitching from arm-swing inertia.
    """
    from mjlab.managers import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from src.policy_walk import joint_deviation_l1
    cfg.rewards["hand_still"] = RewardTermCfg(
        func=joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hand_.*"], preserve_order=False)},
    )


def action_rate_l2_base_only(env) -> "torch.Tensor":
    """L2 action-rate penalty scoped to the first 29 action dims (G1 base joints).
    Args: env: RL env; the raw policy output is indexed before the per-term scale.
    Returns: (num_envs,) tensor = sum_{i<29} (a_i_t - a_i_{t-1})^2; matches policy_walk.action_rate_l2 on 29-DOF but excludes the 14 Dex3 hand dims whose high-stiffness-locked actuators are agnostic to policy output.
    """
    import torch
    a_now = env.action_manager.action[:, :29]
    a_prev = env.action_manager.prev_action[:, :29]
    return torch.sum(torch.square(a_now - a_prev), dim=1)


def _scope_action_rate_to_base(cfg) -> None:
    """Replace the global action_rate_l2 term with the base-29-only variant.
    Args: cfg: env cfg whose action_rate_l2 reward term is overwritten in place.
    Returns: none; prevents the 14 locked hand action dims from inflating the rate penalty and biasing the policy toward all-zero actions.
    """
    term = cfg.rewards.get("action_rate_l2")
    if term is not None:
        term.func = action_rate_l2_base_only


def g1_flat_env_cfg_dex3(play: bool = False):
    """Build the Dex3 variant of the walk env via a one-shot monkey-patch of policy_walk globals.
    Args: play: forwarded to policy_walk.g1_flat_env_cfg to toggle the play-mode overrides.
    Returns: env cfg with the base MDP intact (rewards / sensors / commands / terminations) and robot cfg, joint list and action scale swapped for the 43-DOF body.
    """
    orig_robot = _pw.get_g1_robot_cfg
    orig_joints = _pw.G1_JOINTS
    orig_scale = _pw.G1_ACTION_SCALE
    try:
        _pw.get_g1_robot_cfg = _dex3_robot_cfg
        _pw.G1_JOINTS = _dex3_joint_tuple()
        _pw.G1_ACTION_SCALE = _dex3_action_scale()
        cfg = _pw.g1_flat_env_cfg(play=play)
    finally:
        _pw.get_g1_robot_cfg = orig_robot
        _pw.G1_JOINTS = orig_joints
        _pw.G1_ACTION_SCALE = orig_scale
    _extend_pose_stds_for_hands(cfg)
    _add_hand_still_reward(cfg)
    _scope_action_rate_to_base(cfg)
    return cfg


def g1_ppo_runner_cfg_dex3():
    """Build the PPO runner cfg for the Dex3 variant.
    Args: none.
    Returns: runner cfg with the base 29-DOF walk hyperparams and mirror aug wired through the base-only variant of src.symmetry.mirror_g1_walk_dex3 (14 hand DOF are passed through unchanged so the PPO ratio stays exact on the hand slice and the base L/R symmetry still accelerates leg/arm learning).
    """
    rl = _pw.g1_ppo_runner_cfg()
    rl.experiment_name = "g1_walk_dex3"
    rl.algorithm.symmetry_cfg = {
        "use_data_augmentation": True,
        "use_mirror_loss": False,
        "mirror_loss_coeff": 0.0,
        "data_augmentation_func": "src.symmetry:mirror_g1_walk_dex3",
    }
    return rl


def register():
    """Register the Dex3 walk task with mjlab.
    Args: none; the function drops any prior entry at TASK_ID so the call is idempotent.
    Returns: the TASK_ID string just registered.
    """
    from mjlab.tasks import registry
    from mjlab.tasks.registry import register_mjlab_task
    from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

    registry._REGISTRY.pop(TASK_ID, None)
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=g1_flat_env_cfg_dex3(),
        play_env_cfg=g1_flat_env_cfg_dex3(play=True),
        rl_cfg=g1_ppo_runner_cfg_dex3(),
        runner_cls=VelocityOnPolicyRunner,
    )
    return TASK_ID
