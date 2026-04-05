"""DAgger active correction via action replay.

Finds trust boundary in imagined trajectories and replays actions
in the real environment to collect corrective transitions.
"""

import jax.numpy as jnp
import numpy as np


def find_trust_boundary(cum_trust, tau):
    """Find the first step where cumulative trust drops below tau.

    Args:
        cum_trust: Cumulative trust scores. # (B, H)
        tau: Truncation threshold.

    Returns:
        boundary_indices: Index of first step below tau per trajectory. # (B,)
            Returns H if trust never drops below tau.
    """
    # (B, H) boolean mask: True where trust is below tau
    below_tau = cum_trust < tau
    # For each row, find the first True index; if none, return H
    H = cum_trust.shape[-1]
    # argmax on bool gives first True; if all False, returns 0
    # So we need to distinguish "first step is below" from "none below"
    has_drop = below_tau.any(axis=-1)  # (B,)
    first_below = jnp.argmax(below_tau, axis=-1)  # (B,)
    return jnp.where(has_drop, first_below, H)


def action_replay_correct(env, qpos_0, qvel_0, actions, boundary_k):
    """Replay actions in real env up to boundary_k, collect correct transition.

    Replays actions[0:boundary_k] to reach state s_k, then executes
    actions[boundary_k] to get the real s_{k+1}.

    Args:
        env: MuJoCo environment with set_state(qpos, qvel) and step(action).
        qpos_0: Starting joint positions. # (qpos_dim,)
        qvel_0: Starting joint velocities. # (qvel_dim,)
        actions: Action sequence from imagined trajectory. # (H, act_dim)
        boundary_k: Step index to correct (0-indexed).

    Returns:
        dict with keys:
            obs: observation at s_k
            action: action at step k
            reward: real reward from (s_k, a_k)
            next_obs: observation at s_{k+1}
            qpos: qpos at s_k
            qvel: qvel at s_k
        Returns None if boundary_k >= len(actions).
    """
    H = len(actions)
    if boundary_k >= H:
        return None

    # Convert JAX arrays to numpy for env interaction
    qpos_0 = np.asarray(qpos_0)
    qvel_0 = np.asarray(qvel_0)
    actions = np.asarray(actions)

    # Reset to starting state
    env.set_state(qpos_0, qvel_0)
    obs = env.get_obs() if hasattr(env, 'get_obs') else None

    # Replay actions[0:k] to reach s_k
    for i in range(boundary_k):
        obs, _, _, _ = env.step(actions[i])

    # Record s_k state
    qpos_k = env.get_state()[0] if hasattr(env, 'get_state') else None
    qvel_k = env.get_state()[1] if hasattr(env, 'get_state') else None
    obs_k = obs

    # Execute action at boundary to get real s_{k+1}
    obs_next, reward, done, info = env.step(actions[boundary_k])

    return {
        'obs': obs_k,
        'action': actions[boundary_k],
        'reward': reward,
        'next_obs': obs_next,
        'qpos': qpos_k,
        'qvel': qvel_k,
    }


def collect_corrections(env, qpos_0, qvel_0, actions, cum_trust, tau,
                        max_corrections=1):
    """Collect DAgger corrections for a single imagined trajectory.

    Finds trust boundary and replays to collect corrective data.

    Args:
        env: MuJoCo environment.
        qpos_0: Starting joint positions. # (qpos_dim,)
        qvel_0: Starting joint velocities. # (qvel_dim,)
        actions: Action sequence. # (H, act_dim)
        cum_trust: Cumulative trust scores. # (H,)
        tau: Trust threshold.
        max_corrections: Max number of corrections per trajectory.

    Returns:
        List of correction dicts (from action_replay_correct).
    """
    H = len(actions)
    corrections = []
    below_tau = np.asarray(cum_trust < tau)

    # Find indices where trust first drops below tau
    drop_indices = np.where(below_tau)[0]
    if len(drop_indices) == 0:
        return corrections

    # Correct at most max_corrections boundaries
    for k in drop_indices[:max_corrections]:
        result = action_replay_correct(env, qpos_0, qvel_0, actions, int(k))
        if result is not None:
            corrections.append(result)

    return corrections
