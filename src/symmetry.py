##########################################################################################
# Project    : unitree-g1-humanoid-walk                                                  #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-walk                    #
##########################################################################################
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

# Canonical mjlab and Unitree SDK order for the G1 29-DOF, kept as a tuple so
# it can be diffed against G1_JOINTS at import time (see _assert_joint_order).
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

# Permutation OUT[i] takes value from IN[G1_MIRROR_PERM[i]].
# Legs (L 0-5 vs R 6-11), waist 12-14 stays, arms (L 15-21 vs R 22-28).
perm = list(range(29))
perm[0:6] = list(range(6, 12))
perm[6:12] = list(range(0, 6))
perm[15:22] = list(range(22, 29))
perm[22:29] = list(range(15, 22))
G1_MIRROR_PERM = tuple(perm)
del perm

# Joints whose axis is roll (x) or yaw (z) invert sign under sagittal mirror.
# Pitch axes (y-axis) do NOT invert.
G1_MIRROR_SIGN_FLIP_IDX = (1, 2, 5, 7, 8, 11, 12, 13, 16, 17, 19, 21, 23, 24, 26, 28)


def _assert_joint_order():
    """Verify runtime G1_JOINTS matches the hard-coded G1_29DOF_JOINT_ORDER.
    Args: none.
    Returns: none.
    """
    from src.policy_walk import G1_JOINTS
    if tuple(G1_JOINTS) != G1_29DOF_JOINT_ORDER:
        mismatches = [(i, a, b) for i, (a, b) in enumerate(zip(G1_JOINTS, G1_29DOF_JOINT_ORDER)) if a != b]
        raise AssertionError(
            "G1_JOINTS order in policy_walk differs from src.symmetry G1_29DOF_JOINT_ORDER. "
            f"Mismatches (idx, actual, expected): {mismatches[:5]} ..."
        )


_assert_joint_order()


def _switch_g1_joints_left_right(x: torch.Tensor) -> torch.Tensor:
    """Permute left/right on the last dim of a (..., 29) tensor and flip roll/yaw signs.
    Args: x: tensor whose last dim is exactly 29 (joint_pos, joint_vel or actions).
    Returns: mirrored tensor, same shape.
    """
    perm = torch.as_tensor(G1_MIRROR_PERM, device=x.device, dtype=torch.long)
    out = torch.index_select(x, dim=-1, index=perm)
    signs = torch.ones(29, device=x.device, dtype=x.dtype)
    signs[list(G1_MIRROR_SIGN_FLIP_IDX)] = -1.0
    return out * signs


# Walk obs layout (99 dims); actor and critic share the same terms in policy_walk.build_g1_mdp.

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
    """Mirror a concatenated walk obs tensor, handling history stacking as (..., T*99).
    Args: obs: (..., k*WALK_OBS_DIM) tensor for k >= 1 stacked timesteps.
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
    Args: actions: (..., 29) tensor.
    Returns: mirrored actions, same shape.
    """
    return _switch_g1_joints_left_right(actions)


@torch.no_grad()
def mirror_g1_walk(env=None, obs=None, actions=None):
    """rsl_rl symmetry_cfg entrypoint returning a doubled batch [original; mirrored].
    Args: env: mjlab env (unused, kept for the rsl_rl API contract); obs: TensorDict with per-group keys, or plain tensor; actions: (N, 29) tensor, or None.
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


# Dex3 variant (43 DOF: 29 base + 14 hand).
#
# Active strategy: mirror only the 29 base DOF, leave the 14 hand DOF identical in both
# halves of the augmented batch. Rationale: the hand joints have no mirror-symmetric
# reward signal (hand_still penalises deviation from 0 on both hands but does not couple
# left to right), so forcing full-L/R hand symmetry via PPO ratio makes
# new_log_prob(mirror(hand_action)|mirror(hand_obs)) diverge from old_log_prob(hand_action|hand_obs)
# after a few iters, blowing up the gradient and producing NaN actions.
# Keeping hand obs/action identical in both halves makes ratio[N:]==ratio[:N] exactly for
# the hand slice, so the base L/R symmetry still accelerates learning on legs/arms without
# destabilising the fingers.
#
# The full-43-DOF perm + sign-flip table below is kept for reference and in case a future
# task adds a symmetric hand reward (e.g. bimanual carry with balanced grasp).

DEX3_HAND_JOINT_ORDER = (
    "left_hand_thumb_0_joint",  "left_hand_thumb_1_joint",  "left_hand_thumb_2_joint",
    "left_hand_index_0_joint",  "left_hand_index_1_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint","right_hand_middle_1_joint",
)
G1_43DOF_JOINT_ORDER = G1_29DOF_JOINT_ORDER + DEX3_HAND_JOINT_ORDER

# Full-43-DOF mirror (unused by the active path; see rationale above).
_perm43 = list(G1_MIRROR_PERM) + list(range(36, 43)) + list(range(29, 36))
G1_DEX3_MIRROR_PERM_FULL = tuple(_perm43)
del _perm43
_DEX3_HAND_FLIP = (30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42)
G1_DEX3_MIRROR_SIGN_FLIP_IDX_FULL = G1_MIRROR_SIGN_FLIP_IDX + _DEX3_HAND_FLIP
del _DEX3_HAND_FLIP


def _assert_dex3_joint_order():
    """Verify the Dex3 runtime joint tuple matches G1_43DOF_JOINT_ORDER.
    Args: none; call explicitly from policy_walk_dex3 to avoid import-time coupling with the 29-DOF path.
    Returns: none.
    """
    from src.policy_walk_dex3 import _dex3_joint_tuple
    actual = _dex3_joint_tuple()
    if tuple(actual) != G1_43DOF_JOINT_ORDER:
        mismatches = [(i, a, b) for i, (a, b) in enumerate(zip(actual, G1_43DOF_JOINT_ORDER)) if a != b]
        raise AssertionError(f"Dex3 joint order differs from G1_43DOF_JOINT_ORDER. First mismatches: {mismatches[:5]}")


def _switch_g1_joints_left_right_dex3(x: torch.Tensor) -> torch.Tensor:
    """Mirror the first 29 of 43 DOF, leave the 14 hand DOF unchanged.
    Args: x: tensor whose last dim is exactly 43 (29 base joints plus 14 Dex3 hand joints).
    Returns: tensor with the base-29 slice mirrored (perm + sign flip from _switch_g1_joints_left_right) and the hand-14 slice passed through, same shape.
    """
    base = _switch_g1_joints_left_right(x[..., :29])
    hand = x[..., 29:]
    return torch.cat([base, hand], dim=-1)


def _switch_g1_joints_left_right_dex3_full(x: torch.Tensor) -> torch.Tensor:
    """Full 43-DOF mirror including left-right swap of the Dex3 hand joints.
    Args: x: tensor whose last dim is exactly 43 (29 base joints plus 14 Dex3 hand joints).
    Returns: mirrored tensor using G1_DEX3_MIRROR_PERM_FULL + G1_DEX3_MIRROR_SIGN_FLIP_IDX_FULL (thumb_0 keeps sign, other finger joints flip).
    Use with mirror-loss (use_mirror_loss=True, use_data_augmentation=False): the MSE cost is bounded and smooth, so finger asymmetry at init does not blow up the gradient the way PPO ratio on mirrored hand actions does.
    """
    perm = torch.as_tensor(G1_DEX3_MIRROR_PERM_FULL, device=x.device, dtype=torch.long)
    out = torch.index_select(x, dim=-1, index=perm)
    signs = torch.ones(43, device=x.device, dtype=x.dtype)
    signs[list(G1_DEX3_MIRROR_SIGN_FLIP_IDX_FULL)] = -1.0
    return out * signs


WALK_DEX3_OBS_DIM = 141
WALK_DEX3_OBS_LAYOUT = {
    "base_lin_vel":      (0,   3),
    "base_ang_vel":      (3,   6),
    "projected_gravity": (6,   9),
    "joint_pos":         (9,  52),
    "joint_vel":         (52, 95),
    "actions":           (95, 138),
    "command":           (138, 141),
}
assert WALK_DEX3_OBS_LAYOUT["command"][1] == WALK_DEX3_OBS_DIM


def mirror_walk_dex3_obs(obs: torch.Tensor) -> torch.Tensor:
    """Mirror a concatenated walk-Dex3 obs tensor, handling history stacking as (..., T*141).
    Args: obs: (..., k*WALK_DEX3_OBS_DIM) tensor for k >= 1 stacked timesteps.
    Returns: mirrored tensor, same shape.
    """
    assert obs.shape[-1] % WALK_DEX3_OBS_DIM == 0, (
        f"Last dim {obs.shape[-1]} not multiple of {WALK_DEX3_OBS_DIM}"
    )
    n_stack = obs.shape[-1] // WALK_DEX3_OBS_DIM
    lead = obs.shape[:-1]
    x = obs.clone().view(*lead, n_stack, WALK_DEX3_OBS_DIM)
    for term, (a, b) in WALK_DEX3_OBS_LAYOUT.items():
        if term in WALK_OBS_SIGNS:
            signs = torch.tensor(WALK_OBS_SIGNS[term], device=obs.device, dtype=obs.dtype)
            x[..., a:b] = x[..., a:b] * signs
        else:
            x[..., a:b] = _switch_g1_joints_left_right_dex3(x[..., a:b])
    return x.view(*lead, obs.shape[-1])


def mirror_walk_dex3_actions(actions: torch.Tensor) -> torch.Tensor:
    """Mirror walk-Dex3 actions (43-D joint targets).
    Args: actions: (..., 43) tensor.
    Returns: mirrored actions, same shape.
    """
    return _switch_g1_joints_left_right_dex3(actions)


@torch.no_grad()
def mirror_g1_walk_dex3(env=None, obs=None, actions=None):
    """rsl_rl symmetry_cfg entrypoint for the Dex3 walk variant returning a doubled batch [original; mirrored].
    Args: env: mjlab env (unused, kept for the rsl_rl API contract); obs: TensorDict with per-group keys, or plain tensor; actions: (N, 43) tensor, or None.
    Returns: (obs_out, actions_out); each is None if the corresponding input is None, else doubled along the batch dim.
    Uses the base-only mirror: hand DOFs are passed through identically so the PPO ratio stays exactly 1 on the hand slice. Safe for use_data_augmentation=True.
    """
    obs_out, actions_out = None, None
    if obs is not None:
        if hasattr(obs, "keys") and hasattr(obs, "batch_size"):
            batch_size = obs.batch_size[0]
            obs_out = obs.repeat(2)
            for key in obs.keys():
                obs_out[key][:batch_size] = obs[key]
                obs_out[key][batch_size:] = mirror_walk_dex3_obs(obs[key])
        else:
            obs_mirror = mirror_walk_dex3_obs(obs)
            obs_out = torch.cat([obs, obs_mirror], dim=0)
    if actions is not None:
        actions_mirror = mirror_walk_dex3_actions(actions)
        actions_out = torch.cat([actions, actions_mirror], dim=0)
    return obs_out, actions_out


def mirror_walk_dex3_obs_full(obs: torch.Tensor) -> torch.Tensor:
    """Full-43-DOF mirror of a concatenated walk-Dex3 obs tensor (hands included).
    Args: obs: (..., k*WALK_DEX3_OBS_DIM) tensor for k >= 1 stacked timesteps.
    Returns: mirrored tensor, same shape, with left-right hand swap and finger sign flips (per _switch_g1_joints_left_right_dex3_full).
    """
    assert obs.shape[-1] % WALK_DEX3_OBS_DIM == 0, (
        f"Last dim {obs.shape[-1]} not multiple of {WALK_DEX3_OBS_DIM}"
    )
    n_stack = obs.shape[-1] // WALK_DEX3_OBS_DIM
    lead = obs.shape[:-1]
    x = obs.clone().view(*lead, n_stack, WALK_DEX3_OBS_DIM)
    for term, (a, b) in WALK_DEX3_OBS_LAYOUT.items():
        if term in WALK_OBS_SIGNS:
            signs = torch.tensor(WALK_OBS_SIGNS[term], device=obs.device, dtype=obs.dtype)
            x[..., a:b] = x[..., a:b] * signs
        else:
            x[..., a:b] = _switch_g1_joints_left_right_dex3_full(x[..., a:b])
    return x.view(*lead, obs.shape[-1])


def mirror_walk_dex3_actions_full(actions: torch.Tensor) -> torch.Tensor:
    """Full-43-DOF mirror of walk-Dex3 actions (hands included).
    Args: actions: (..., 43) tensor.
    Returns: mirrored actions, same shape.
    """
    return _switch_g1_joints_left_right_dex3_full(actions)


@torch.no_grad()
def mirror_g1_walk_dex3_full(env=None, obs=None, actions=None):
    """Full-43-DOF mirror entrypoint intended for use_mirror_loss=True (not use_data_augmentation).
    Args: env: unused; obs: TensorDict or tensor; actions: (N, 43) tensor or None.
    Returns: (obs_out, actions_out) doubled along the batch dim; use_mirror_loss builds a bounded MSE term between actor(mirror(obs)) and mirror(actor(obs)), so finger asymmetry at init drives a gradient without the PPO-ratio blow-up seen with use_data_augmentation on the full 43-DOF mirror.
    """
    obs_out, actions_out = None, None
    if obs is not None:
        if hasattr(obs, "keys") and hasattr(obs, "batch_size"):
            batch_size = obs.batch_size[0]
            obs_out = obs.repeat(2)
            for key in obs.keys():
                obs_out[key][:batch_size] = obs[key]
                obs_out[key][batch_size:] = mirror_walk_dex3_obs_full(obs[key])
        else:
            obs_mirror = mirror_walk_dex3_obs_full(obs)
            obs_out = torch.cat([obs, obs_mirror], dim=0)
    if actions is not None:
        actions_mirror = mirror_walk_dex3_actions_full(actions)
        actions_out = torch.cat([actions, actions_mirror], dim=0)
    return obs_out, actions_out
