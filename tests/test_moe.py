"""Unit tests for MoE Dynamics module."""

import jax
import jax.numpy as jnp
import ninjax as nj
import pytest
import embodied.jax.nets as nn

import sys
sys.path.insert(0, '/Users/chengcheng/grounded_imagination')

from grounded.moe_dynamics import MoECore


def _make_moe(num_experts=4, deter=128, hidden=64, blocks=4, dynlayers=1):
    return MoECore(
        name='moe_test',
        deter=deter, hidden=hidden, blocks=blocks,
        dynlayers=dynlayers, num_experts=num_experts,
        norm='rms', act='gelu', balance_coef=0.01,
    )


def _dummy_inputs(batch=8, deter=128, hidden=64):
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key, 2)
    deter_state = jax.random.normal(k1, (batch, deter))
    preproc = jax.random.normal(k2, (batch, hidden * 3))
    return nn.cast((deter_state, preproc))


class TestMoECore:

    def test_output_shape(self):
        B, D, H = 8, 128, 64
        moe = _make_moe(deter=D, hidden=H)
        deter, preproc = _dummy_inputs(B, D, H)

        @nj.pure
        def run(moe, deter, preproc):
            return moe(deter, preproc)

        state = {}
        state, (new_deter, router_w) = run(
            state, moe, deter, preproc,
            seed=jax.random.PRNGKey(42), create=True)
        assert new_deter.shape == (B, D)
        assert router_w.shape == (B, 4)

    def test_top2_routing(self):
        B, D, H = 8, 128, 64
        moe = _make_moe(deter=D, hidden=H)
        deter, preproc = _dummy_inputs(B, D, H)

        @nj.pure
        def run(moe, deter, preproc):
            _, router_weights = moe(deter, preproc)
            return router_weights

        state = {}
        state, router_weights = run(
            state, moe, deter, preproc,
            seed=jax.random.PRNGKey(42), create=True)
        row_sums = router_weights.sum(axis=-1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-2), \
            f"Router weights should sum to ~1, got {row_sums}"

    def test_load_balance_loss(self):
        B, D, H = 8, 128, 64
        moe = _make_moe(deter=D, hidden=H)
        deter, preproc = _dummy_inputs(B, D, H)

        @nj.pure
        def run(moe, deter, preproc):
            _, router_weights = moe(deter, preproc)
            return MoECore.compute_balance_loss(router_weights, 0.01)

        state = {}
        state, loss = run(
            state, moe, deter, preproc,
            seed=jax.random.PRNGKey(42), create=True)
        assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
        assert float(loss) >= 0

    def test_different_experts_different_outputs(self):
        B, D, H = 4, 128, 64
        moe = _make_moe(num_experts=4, deter=D, hidden=H)
        deter, preproc = _dummy_inputs(B, D, H)

        @nj.pure
        def run(moe, deter, preproc):
            _ = moe(deter, preproc)
            g = moe.blocks
            preproc_grouped = preproc[..., None, :].repeat(g, -2)
            outputs = [
                moe._expert(f'expert{k}', deter, preproc_grouped)
                for k in range(4)
            ]
            return jnp.stack(outputs, axis=0)  # (4, B, D)

        state = {}
        state, expert_outputs = run(
            state, moe, deter, preproc,
            seed=jax.random.PRNGKey(42), create=True)
        diff_01 = jnp.abs(expert_outputs[0] - expert_outputs[1]).mean()
        assert float(diff_01) > 1e-6, "Experts should differ after init"

    def test_backward(self):
        B, D, H = 4, 128, 64
        moe = _make_moe(deter=D, hidden=H)
        deter, preproc = _dummy_inputs(B, D, H)

        @nj.pure
        def run(moe, deter, preproc):
            new_deter, _ = moe(deter, preproc)
            return new_deter

        state = {}
        state, _ = run(
            state, moe, deter, preproc,
            seed=jax.random.PRNGKey(42), create=True)

        def loss_fn(state):
            state, out = run(
                state, moe, deter, preproc,
                seed=jax.random.PRNGKey(42), create=True)
            return out.mean(), state

        grads, _ = jax.grad(loss_fn, has_aux=True)(state)
        total_grad = sum(jax.tree.leaves(
            jax.tree.map(lambda x: jnp.abs(x).sum(), grads)))
        assert float(total_grad) > 0, "Gradients should be non-zero"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
