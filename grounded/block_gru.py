"""Shared Block GRU step function.

Used by both RSSM._core (vanilla) and MoECore._expert (MoE).
Eliminates code duplication between rssm.py and moe_dynamics.py.
"""

import einops
import jax
import jax.numpy as jnp
import embodied.jax.nets as nn


def block_gru_step(module, name, deter, preproc_grouped, blocks, deter_dim,
                   dynlayers, act, norm, **kw):
    """Single Block GRU dynamics step.

    Args:
        module: nj.Module to call self.sub() on.
        name: Name prefix for sub-modules.
        deter: Previous deterministic state. # (B, deter_dim)
        preproc_grouped: Preproc repeated per group. # (B, g, hidden*3)
        blocks: Number of block groups.
        deter_dim: Deterministic state dimension.
        dynlayers: Number of hidden layers.
        act: Activation function name.
        norm: Normalization type.

    Returns:
        new_deter: Updated deterministic state. # (B, deter_dim)
    """
    g = blocks
    flat2group = lambda x: einops.rearrange(x, '... (g h) -> ... g h', g=g)
    group2flat = lambda x: einops.rearrange(x, '... g h -> ... (g h)', g=g)

    x = group2flat(jnp.concatenate([flat2group(deter), preproc_grouped], -1))

    for i in range(dynlayers):
        x = module.sub(f'{name}hid{i}', nn.BlockLinear, deter_dim, g, **kw)(x)
        x = nn.act(act)(module.sub(f'{name}hid{i}n', nn.Norm, norm)(x))

    x = module.sub(f'{name}gru', nn.BlockLinear, 3 * deter_dim, g, **kw)(x)
    gates = jnp.split(flat2group(x), 3, -1)
    reset, cand, update = [group2flat(g_) for g_ in gates]
    reset = jax.nn.sigmoid(reset)
    cand = jnp.tanh(reset * cand)
    update = jax.nn.sigmoid(update - 1)
    return update * cand + (1 - update) * deter
