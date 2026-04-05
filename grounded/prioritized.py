"""Prioritized experience replay utilities.

Provides priority computation from TRD scores, importance sampling
weight correction, and beta annealing schedule.
"""

import jax.numpy as jnp


def compute_priorities(trd_scores):
    """Convert TRD scores to replay priorities.

    Lower TRD score = world model less accurate = higher priority.

    Args:
        trd_scores: TRD trust scores in (0, 1). # (B,)

    Returns:
        priorities: Replay priorities in (0, 1]. # (B,)
    """
    return 1.0 - trd_scores + 1e-6  # small epsilon to avoid zero priority


def importance_sampling_weights(priorities, alpha=0.6, beta=0.4):
    """Compute importance sampling weights to correct prioritized sampling bias.

    Args:
        priorities: Per-sample priorities. # (B,)
        alpha: Priority exponent (0=uniform, 1=full priority).
        beta: IS correction exponent (0=no correction, 1=full correction).

    Returns:
        weights: Normalized IS weights. # (B,)
    """
    N = priorities.shape[0]
    # Sampling probabilities: P(i) = p_i^alpha / sum(p_j^alpha)
    probs = priorities ** alpha
    probs = probs / (probs.sum() + 1e-8)  # (B,)
    # IS weights: w_i = (N * P(i))^(-beta)
    weights = (N * probs) ** (-beta)  # (B,)
    # Normalize by max for stability
    weights = weights / (weights.max() + 1e-8)  # (B,)
    return weights


def beta_schedule(step, total_steps, beta_start=0.4, beta_end=1.0):
    """Linear annealing schedule for IS correction exponent beta.

    Args:
        step: Current training step.
        total_steps: Total training steps.
        beta_start: Initial beta (less correction, faster learning).
        beta_end: Final beta (full correction, unbiased).

    Returns:
        Current beta value.
    """
    frac = jnp.clip(step / jnp.maximum(total_steps, 1), 0.0, 1.0)
    return beta_start + frac * (beta_end - beta_start)
