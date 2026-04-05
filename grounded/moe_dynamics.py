"""MoE Dynamics module for RSSM.

Replaces the single Block GRU dynamics in RSSM._core with
multiple homogeneous experts + Top-2 soft routing.
"""

import einops
import jax
import jax.numpy as jnp
import ninjax as nj
import embodied.jax.nets as nn

f32 = jnp.float32


class MoECore(nj.Module):
    """Mixture-of-Experts dynamics core for RSSM.

    Each expert is a Block GRU with identical structure but different
    parameters (different random initialization). A learned router
    selects Top-2 experts per input and combines their outputs.
    """

    deter: int = 4096
    hidden: int = 2048
    blocks: int = 8
    dynlayers: int = 1
    num_experts: int = 4
    norm: str = 'rms'
    act: str = 'gelu'
    balance_coef: float = 0.01

    def __init__(self, **kw):
        self.kw = kw

    def __call__(self, deter, preproc):
        """Run MoE dynamics.

        Args:
            deter: Previous deterministic state. # (B, deter)
            preproc: Preprocessed concatenated input. # (B, hidden*3)

        Returns:
            new_deter: Updated deterministic state. # (B, deter)
            router_weights: Full router softmax output. # (B, num_experts)
        """
        g = self.blocks

        # Router
        x_r = self.sub('router0', nn.Linear, 128, **self.kw)(preproc)
        x_r = jax.nn.gelu(x_r)  # (B, 128)
        router_logits = self.sub(
            'router1', nn.Linear, self.num_experts, **self.kw)(x_r)
        router_weights = jax.nn.softmax(router_logits, axis=-1)  # (B, num_experts)

        # Top-2 selection and renormalization
        top2_weights, top2_indices = jax.lax.top_k(router_weights, 2)
        top2_weights = top2_weights / (
            top2_weights.sum(axis=-1, keepdims=True) + 1e-8)  # (B, 2)

        # Precompute shared input for all experts
        preproc_grouped = preproc[..., None, :].repeat(g, -2)  # (B, g, hidden*3)

        # Compute all experts (JAX JIT compiles all paths; no savings from sparse dispatch)
        expert_outputs = jnp.stack([
            self._expert(f'expert{k}', deter, preproc_grouped)
            for k in range(self.num_experts)
        ], axis=0)  # (num_experts, B, deter)

        # Gather Top-2 and combine
        B = deter.shape[0]
        batch_idx = jnp.arange(B)
        out0 = expert_outputs[top2_indices[:, 0], batch_idx]  # (B, deter)
        out1 = expert_outputs[top2_indices[:, 1], batch_idx]  # (B, deter)
        new_deter = top2_weights[:, 0:1] * out0 + top2_weights[:, 1:2] * out1

        return new_deter, router_weights

    def _expert(self, name, deter, preproc_grouped):
        """Single Block GRU expert.

        Args:
            name: Unique name prefix for this expert's parameters.
            deter: Previous deterministic state. # (B, deter)
            preproc_grouped: Precomputed grouped preproc. # (B, g, hidden*3)

        Returns:
            new_deter: Updated deterministic state. # (B, deter)
        """
        g = self.blocks
        flat2group = lambda x: einops.rearrange(x, '... (g h) -> ... g h', g=g)
        group2flat = lambda x: einops.rearrange(x, '... g h -> ... (g h)', g=g)

        x = group2flat(jnp.concatenate([flat2group(deter), preproc_grouped], -1))

        for i in range(self.dynlayers):
            x = self.sub(f'{name}_hid{i}', nn.BlockLinear,
                         self.deter, g, **self.kw)(x)
            x = nn.act(self.act)(
                self.sub(f'{name}_hid{i}n', nn.Norm, self.norm)(x))

        x = self.sub(f'{name}_gru', nn.BlockLinear,
                     3 * self.deter, g, **self.kw)(x)
        gates = jnp.split(flat2group(x), 3, -1)
        reset, cand, update = [group2flat(g_) for g_ in gates]
        reset = jax.nn.sigmoid(reset)
        cand = jnp.tanh(reset * cand)
        update = jax.nn.sigmoid(update - 1)
        return update * cand + (1 - update) * deter  # (B, deter)

    @staticmethod
    def compute_balance_loss(router_weights, coef=0.01):
        """Compute load balancing loss: coef * CV(load)^2.

        Args:
            router_weights: Router softmax output. # (B, num_experts)
            coef: Balance loss coefficient.

        Returns:
            Scalar loss value >= 0.
        """
        load = router_weights.mean(axis=0)  # (num_experts,)
        cv = load.std() / (load.mean() + 1e-8)
        return coef * cv ** 2
