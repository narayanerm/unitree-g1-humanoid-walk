###############################################################################
# Project    : Humanoid Loco-Manipulation with the Unitree G1                 #
# Developer  : Narayane Ribeiro Medeiros                                      #
# Repository : https://github.com/narayanerm/Humanoid-Loco-Manipulation       #
###############################################################################
"""Left-right symmetry (sagittal-plane mirror) for the G1 walk task.

Input:  obs tensor (or TensorDict with actor/critic keys) and actions tensor.
Output: doubled batch [original; mirrored] used by rsl_rl PPO with use_data_augmentation=True.

Physics of reflection across the XZ plane (y flips):
- regular vectors  v -> (v_x, -v_y,  v_z)              sign = [ 1, -1,  1]
- pseudo-vectors  omega -> (-omega_x,  omega_y, -omega_z)  sign = [-1,  1, -1]
- twist command (v_x, v_y, w_z) -> (v_x, -v_y, -w_z)   sign = [ 1, -1, -1]

The joint permutation and sign table target Unitree G1 29-DOF in SDK/mjlab
canonical order (verified to match at import time).

Adapted from https://github.com/mintlabkorea/unitree_g1_rl_lab
(symmetry_g1.py, BSD-3-Clause). Fixes the base_ang_vel sign vector that
upstream had wrong ([1, -1, -1] -> [-1, 1, -1]) per IsaacLab's ANYmal
reference implementation and standard pseudo-vector transformation rules.
"""

from __future__ import annotations

import torch

# ==================== joint mapping ====================

# Canonical mjlab and Unitree SDK order for the G1 29-DOF. Kept as a tuple so
# it can be diff'd against G1_JOINTS at import time (see _assert_joint_order).
G1_29DOF_JOINT_ORDER = (
    "left_hip_pitch_joint",  "left_hip_roll_joint",  "left_hip_yaw_joint",
    "left_knee_joint",       "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint",      "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",       "waist_roll_joint",     "waist_pitch_joint",
    "left_shoulder_pitch_joint",  "left_shoulder_roll_joint",  "left_shoulder_yaw_joint",
    "left_elbow_joint",           "left_wrist_roll_joint",     "left_wrist_pitch_joint",  "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint",          "right_wrist_roll_joint",    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

# Permutation: OUT[i] takes value from IN[G1_MIRROR_PERM[i]].
# Legs (L 0-5 <-> R 6-11), waist 12-14 stays, arms (L 15-21 <-> R 22-28).
_perm = list(range(29))
_perm[0:6] = list(range(6, 12))
_perm[6:12] = list(range(0, 6))
_perm[15:22] = list(range(22, 29))
_perm[22:29] = list(range(15, 22))
G1_MIRROR_PERM = tuple(_perm)

# Joints whose axis is roll (x) or yaw (z) invert sign under sagittal mirror.
# Pitch axes (y-axis) do NOT invert.
G1_MIRROR_SIGN_FLIP_IDX = (1, 2, 5, 7, 8, 11, 12, 13, 16, 17, 19, 21, 23, 24, 26, 28)


def _assert_joint_order():
    """Verify runtime G1_JOINTS matches the hard-coded G1_29DOF_JOINT_ORDER."""
    from src.policy_walk import G1_JOINTS
    if tuple(G1_JOINTS) != G1_29DOF_JOINT_ORDER:
        mismatches = [(i, a, b) for i, (a, b) in enumerate(zip(G1_JOINTS, G1_29DOF_JOINT_ORDER)) if a != b]
        raise AssertionError(
            "G1_JOINTS order in policy_walk differs from src.symmetry G1_29DOF_JOINT_ORDER. "
            f"Mismatches (idx, actual, expected): {mismatches[:5]} ..."
        )


_assert_joint_order()


def _switch_g1_joints_left_right(x: torch.Tensor) -> torch.Tensor:
    """Permute L<->R on the last dim of a (..., 29) tensor and flip roll/yaw signs.
    Args: x — tensor whose last dim is exactly 29 (joint_pos, joint_vel or actions).
    Returns: mirrored tensor, same shape.
    """
    perm = torch.as_tensor(G1_MIRROR_PERM, device=x.device, dtype=torch.long)
    out = torch.index_select(x, dim=-1, index=perm)
    signs = torch.ones(29, device=x.device, dtype=x.dtype)
    signs[list(G1_MIRROR_SIGN_FLIP_IDX)] = -1.0
    return out * signs


# ==================== walk obs layout (99 dims) ====================
# actor and critic share the same terms in policy_walk.build_g1_mdp.

WALK_OBS_DIM = 99

WALK_OBS_LAYOUT = {
    "base_lin_vel":      (0, 3),
    "base_ang_vel":      (3, 6),
    "projected_gravity": (6, 9),
    "joint_pos":         (9, 38),
    "joint_vel":         (38, 67),
    "actions":           (67, 96),
    "command":           (96, 99),
}
# Sanity check the layout adds up.
assert WALK_OBS_LAYOUT["command"][1] == WALK_OBS_DIM

WALK_OBS_SIGNS = {
    "base_lin_vel":      (1.0, -1.0,  1.0),
    "base_ang_vel":      (-1.0, 1.0, -1.0),
    "projected_gravity": (1.0, -1.0,  1.0),
    "command":           (1.0, -1.0, -1.0),
}


def mirror_walk_obs(obs: torch.Tensor) -> torch.Tensor:
    """Mirror a concatenated walk obs tensor. Handles history stacking as (..., T*99).
    Args: obs — (..., k*WALK_OBS_DIM) tensor for k >= 1 stacked timesteps.
    Returns: mirrored tensor, same shape.
    """
    assert obs.shape[-1] % WALK_OBS_DIM == 0, (
        f"Last dim {obs.shape[-1]} not multiple of {WALK_OBS_DIM}"
    )
    n_stack = obs.shape[-1] // WALK_OBS_DIM
    lead = obs.shape[:-1]
    x = obs.clone().view(*lead, n_stack, WALK_OBS_DIM)
    for term, (a, b) in WALK_OBS_LAYOUT.items():
        if term in WALK_OBS_SIGNS:
            signs = torch.tensor(WALK_OBS_SIGNS[term], device=obs.device, dtype=obs.dtype)
            x[..., a:b] = x[..., a:b] * signs
        else:
            x[..., a:b] = _switch_g1_joints_left_right(x[..., a:b])
    return x.view(*lead, obs.shape[-1])


def mirror_walk_actions(actions: torch.Tensor) -> torch.Tensor:
    """Mirror walk actions (29-D joint targets).
    Args: actions — (..., 29) tensor.
    Returns: mirrored actions, same shape.
    """
    return _switch_g1_joints_left_right(actions)


# ==================== rsl_rl entrypoint ====================

@torch.no_grad()
def mirror_g1_walk(env=None, obs=None, actions=None):
    """rsl_rl symmetry_cfg entrypoint. Returns doubled batch [original; mirrored].
    Args: env — mjlab env (unused, kept for the rsl_rl API contract). obs — TensorDict with per-group keys, or plain tensor. actions — (N, 29) tensor, or None.
    Returns: (obs_out, actions_out); each is None if the corresponding input is None, else doubled along the batch dim.
    """
    obs_out = None
    actions_out = None

    if obs is not None:
        if hasattr(obs, "keys") and hasattr(obs, "batch_size"):
            batch_size = obs.batch_size[0]
            obs_out = obs.repeat(2)
            for key in obs.keys():
                obs_out[key][:batch_size] = obs[key]
                obs_out[key][batch_size:] = mirror_walk_obs(obs[key])
        else:
            obs_mirror = mirror_walk_obs(obs)
            obs_out = torch.cat([obs, obs_mirror], dim=0)

    if actions is not None:
        actions_mirror = mirror_walk_actions(actions)
        actions_out = torch.cat([actions, actions_mirror], dim=0)

    return obs_out, actions_out
