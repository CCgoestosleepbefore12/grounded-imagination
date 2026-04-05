"""Diagnostic tools for Grounded Imagination framework.

Analysis functions for paper visualizations (Figures 1-5):
- Prediction error vs rollout step
- TRD scores over episode
- MoE router weight analysis
- Router entropy
- Training metrics summary
"""

import jax
import jax.numpy as jnp
import numpy as np


def prediction_error_by_step(imagine_fn, feat2tensor, starts, actions,
                             real_feats, max_steps=20):
    """Compute world model prediction MSE at each rollout step.

    Imagines from real starting states and compares with ground truth
    at each step. Useful for Figure 1 (motivation).

    Args:
        imagine_fn: RSSM.imagine function (carry, actions, length, training).
        feat2tensor: Function to convert feat dict to flat tensor.
        starts: Starting carry states from replay. # dict with (B, ...)
        actions: Real action sequence. # dict with (B, H, act_dim)
        real_feats: Ground truth feat sequence. # dict with (B, H, ...)
        max_steps: Maximum rollout steps to evaluate.

    Returns:
        mse_per_step: MSE at each step. # (max_steps,)
    """
    H = min(max_steps, jax.tree.leaves(actions)[0].shape[1])
    _, imgfeat, _ = imagine_fn(starts, actions, H, training=False)
    z_pred = feat2tensor(imgfeat)  # (B, H, z_dim)
    z_real = feat2tensor(
        jax.tree.map(lambda x: x[:, :H], real_feats))  # (B, H, z_dim)
    # Per-step MSE averaged over batch and z_dim
    mse = ((z_pred - z_real) ** 2).mean(axis=(0, -1))  # (H,)
    return mse


def trd_scores_over_episode(trd, feat_tensor, actions):
    """Compute TRD trust scores for each transition in an episode.

    Useful for Figure 2 (TRD scores + effective imagination steps).

    Args:
        trd: TRD module (callable).
        feat_tensor: Flattened feat tensor for one episode. # (T, z_dim)
        actions: Action tensor for the episode. # (T, a_dim)

    Returns:
        per_step_trust: TRD score per transition. # (T-1,)
        cumulative_trust: Cumulative product of trust scores. # (T-1,)
    """
    z_t = feat_tensor[:-1]  # (T-1, z_dim)
    z_next = feat_tensor[1:]  # (T-1, z_dim)
    a_t = actions[:-1]  # (T-1, a_dim)
    per_step_trust = trd(z_t, a_t, z_next)  # (T-1,)
    cumulative_trust = jnp.cumprod(per_step_trust)  # (T-1,)
    return per_step_trust, cumulative_trust


def router_weights_over_episode(moe_core, deter_seq, preproc_seq):
    """Extract MoE router weights for each timestep in an episode.

    Useful for Figure 3 (expert usage over time).

    Args:
        moe_core: MoECore module (must have router submodules).
        deter_seq: Deterministic states. # (T, deter_dim)
        preproc_seq: Preprocessed inputs. # (T, hidden*3)

    Returns:
        weights: Router softmax weights per timestep. # (T, num_experts)
    """
    _, router_weights = moe_core(deter_seq, preproc_seq)  # (T, num_experts)
    return router_weights


def router_entropy(router_weights):
    """Compute entropy of router weight distribution.

    Higher entropy = more uniform expert usage.
    Maximum entropy = log(num_experts) when all experts used equally.

    Args:
        router_weights: Router softmax output. # (B, num_experts) or (T, num_experts)

    Returns:
        entropy: Mean entropy across batch/time. Scalar.
    """
    eps = 1e-8
    ent = -(router_weights * jnp.log(router_weights + eps)).sum(axis=-1)  # (B,)
    return ent.mean()


def effective_imagination_steps(cumulative_trust, tau):
    """Count how many imagination steps have trust above threshold.

    Useful for tracking self-improvement over training (Figure 4).

    Args:
        cumulative_trust: Cumulative trust scores. # (B, H)
        tau: Trust threshold.

    Returns:
        mean_steps: Average effective steps per trajectory. Scalar.
    """
    return (cumulative_trust > tau).sum(axis=-1).astype(jnp.float32).mean()


def training_metrics_summary(metrics_history):
    """Summarize training metrics for convergence plots (Figure 4).

    Args:
        metrics_history: List of dicts, one per logging step. Each dict
            may contain: trd_score_real, trd_score_fake, imag_trust_mean,
            imag_effective_steps, episode_return, etc.

    Returns:
        summary: Dict of numpy arrays, one per metric, indexed by step.
    """
    if not metrics_history:
        return {}
    all_keys = set()
    for m in metrics_history:
        all_keys.update(m.keys())
    summary = {}
    for key in sorted(all_keys):
        values = [float(m[key]) for m in metrics_history if key in m]
        summary[key] = np.array(values)
    return summary
