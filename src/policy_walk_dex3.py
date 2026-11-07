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
DEX3_HAND_STIFFNESS = 5.0
DEX3_HAND_DAMPING   = 0.5
DEX3_HAND_EFFORT    = 2.5
DEX3_HAND_ARMATURE  = 1.0e-4

G1_DEX3_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hand_(thumb|index|middle)_[0-2]_joint",),
    stiffness=DEX3_HAND_STIFFNESS,
    damping=DEX3_HAND_DAMPING,
    effort_limit=DEX3_HAND_EFFORT,
    armature=DEX3_HAND_ARMATURE,
)


def _dex3_robot_cfg() -> EntityCfg:
    """Build the EntityCfg for the G1 + Dex3 body.
    Args: none.
    Returns: EntityCfg combining the base G1 articulation with the Dex3 hand actuator block; collision uses FULL_COLLISION_WITHOUT_SELF so the 14 finger capsule geoms never contact the robot's own torso/arms (unstable at 8192 envs with FULL_COLLISION, produced NaN velocities in the first step).
    """
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ARTICULATION, FULL_COLLISION_WITHOUT_SELF
    articulation = EntityArticulationInfoCfg(
        actuators=tuple(G1_ARTICULATION.actuators) + (G1_DEX3_ACTUATOR,),
        soft_joint_pos_limit_factor=G1_ARTICULATION.soft_joint_pos_limit_factor,
    )
    return EntityCfg(
        init_state=KNEES_BENT_KEYFRAME,
        collisions=(FULL_COLLISION_WITHOUT_SELF,),
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
    return cfg


def g1_ppo_runner_cfg_dex3():
    """Build the PPO runner cfg for the Dex3 variant.
    Args: none.
    Returns: runner cfg with the base 29-DOF walk hyperparams and mirror aug wired through the 43-DOF permutation and sign-flip table in src/symmetry.py.
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
