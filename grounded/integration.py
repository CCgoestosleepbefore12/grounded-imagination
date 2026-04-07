"""Integration helpers for Grounded Imagination in DreamerV3.

Extracts grounded-specific computations from agent.py for cleanliness.
All functions are pure (no side effects) and JIT-compatible.
"""

import jax
import jax.numpy as jnp
import embodied.jax.nets as nn

sg = jax.lax.stop_gradient


def compute_trd_loss(trd, feat2tensor, act_space, repfeat, prevact,
                     dyn_extras, B, T):
    """Compute TRD training loss on observed transitions.

    Args:
        trd: TRD module.
        feat2tensor: Function to convert feat dict to flat tensor.
        act_space: Action space dict.
        repfeat: Posterior feat from RSSM observe. # dict with (B, T, ...)
        prevact: Previous actions. # dict with (B, T, act_dim)
        dyn_extras: Extras from RSSM loss containing prior_logit.
        B, T: Batch size and sequence length.

    Returns:
        trd_loss: Scalar loss broadcast to (B, T).
        scores_real: Per-transition real scores. # (B*(T-1),)
        metrics: Dict with trd_score_real, trd_score_fake.
    """
    from grounded.trd import TRD as TRDClass

    z = sg(feat2tensor(repfeat))  # (B, T, z_dim)
    z_t = z[:, :-1]  # (B, T-1, z_dim)
    z_next_real = z[:, 1:]  # (B, T-1, z_dim)
    a_t = nn.cast(nn.DictConcat(act_space, 1)(prevact)[:, 1:])

    # Fake: prior stoch (model prediction without observation)
    prior_stoch = nn.cast(jax.nn.softmax(dyn_extras['prior_logit'], axis=-1))
    z_fake = sg(feat2tensor(dict(deter=repfeat['deter'], stoch=prior_stoch)))
    z_next_pred = z_fake[:, 1:]

    # Flatten and score
    BT1 = B * (T - 1)
    z_t_f = z_t.reshape(BT1, -1)
    a_t_f = a_t.reshape(BT1, -1)
    scores_real = trd(z_t_f, a_t_f, z_next_real.reshape(BT1, -1))
    scores_fake = trd(z_t_f, a_t_f, z_next_pred.reshape(BT1, -1))

    trd_loss = TRDClass.train_loss(scores_real, scores_fake)
    metrics = {
        'trd_score_real': scores_real.mean(),
        'trd_score_fake': scores_fake.mean(),
    }
    return jnp.full((B, T), trd_loss), scores_real, metrics


def compute_trust_weights(trd, feat2tensor, act_space, inp, imgact,
                          gamma, tau):
    """Compute discounted cumulative trust weights for imagination.

    Args:
        trd: TRD module.
        feat2tensor: (unused, inp is already a tensor).
        act_space: Action space dict.
        inp: Imagined feat tensor. # (B*K, H+1, z_dim)
        imgact: Imagined actions. # dict with (B*K, H+1, act_dim)
        gamma: Trust discount factor.
        tau: Trust threshold for effective steps metric.

    Returns:
        imag_trust: Trust weights. # (B*K, H+1)
        metrics: Dict with trust statistics.
    """
    z_img = sg(inp)
    z_t = z_img[:, :-1]  # (B*K, H, z_dim)
    z_next = z_img[:, 1:]  # (B*K, H, z_dim)
    a_img = nn.cast(nn.DictConcat(act_space, 1)(imgact)[:, :-1])
    BK, H = z_t.shape[:2]

    # Per-step trust scores
    trust = trd(
        z_t.reshape(BK * H, -1),
        a_img.reshape(BK * H, -1),
        z_next.reshape(BK * H, -1),
    ).reshape(BK, H)

    # Discounted cumulative trust in log space
    log_trust = jnp.log(trust + 1e-8)

    def _scan(log_cum_prev, log_t):
        log_cum = log_t + gamma * log_cum_prev
        return log_cum, log_cum

    _, log_cum = jax.lax.scan(
        _scan,
        jnp.zeros(BK, dtype=log_trust.dtype),
        log_trust.transpose(1, 0))
    cum_trust = jnp.exp(log_cum.transpose(1, 0))  # (B*K, H)

    # Pad with 1.0 at start (first step has full trust)
    imag_trust = jnp.concatenate([
        jnp.ones((BK, 1), dtype=cum_trust.dtype), cum_trust], axis=1)

    metrics = {
        'imag_trust_mean': imag_trust.mean(),
        'imag_trust_min': imag_trust.min(),
        'imag_effective_steps': (imag_trust > tau).sum(1).mean(),
    }
    return imag_trust, metrics
