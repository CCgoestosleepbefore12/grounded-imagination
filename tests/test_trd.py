"""Unit tests for Transition Realism Discriminator."""

import jax
import jax.numpy as jnp
import ninjax as nj
import pytest
import embodied.jax.nets as nn

import sys
sys.path.insert(0, '/Users/chengcheng/grounded_imagination')

from grounded.trd import TRD, SpectralNormLinear


def _dummy_transition(batch=8, z_dim=128, a_dim=6):
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    z_t = jax.random.normal(k1, (batch, z_dim))
    a_t = jax.random.normal(k2, (batch, a_dim))
    z_next_real = jax.random.normal(k3, (batch, z_dim))
    z_next_fake = jax.random.normal(k4, (batch, z_dim))
    return nn.cast((z_t, a_t, z_next_real, z_next_fake))


class TestTRD:

    def test_output_range(self):
        """TRD output should be in (0, 1)."""
        B = 8
        trd = TRD(name='trd_test', hidden=64)
        z_t, a_t, z_next_real, _ = _dummy_transition(B)

        @nj.pure
        def run(trd, z_t, a_t, z_next):
            return trd(z_t, a_t, z_next)

        state = {}
        state, scores = run(state, trd, z_t, a_t, z_next_real,
                            seed=jax.random.PRNGKey(42), create=True)
        assert scores.shape == (B,)
        assert jnp.all(scores > 0) and jnp.all(scores < 1), \
            f"Scores should be in (0,1), got min={float(scores.min())}, max={float(scores.max())}"

    def test_loss_shape(self):
        """Train loss should be a scalar."""
        B = 8
        trd = TRD(name='trd_test', hidden=64)
        z_t, a_t, z_next_real, z_next_fake = _dummy_transition(B)

        @nj.pure
        def run(trd, z_t, a_t, z_real, z_fake):
            scores_real = trd(z_t, a_t, z_real)
            scores_fake = trd(z_t, a_t, z_fake)
            return TRD.train_loss(scores_real, scores_fake)

        state = {}
        state, loss = run(state, trd, z_t, a_t, z_next_real, z_next_fake,
                          seed=jax.random.PRNGKey(42), create=True)
        assert loss.shape == (), f"Expected scalar, got {loss.shape}"
        assert jnp.isfinite(loss), f"Loss should be finite, got {float(loss)}"

    def test_label_smoothing(self):
        """With label_smooth=0.9, loss should differ from label_smooth=1.0."""
        B = 8
        trd = TRD(name='trd_test', hidden=64)
        z_t, a_t, z_next_real, z_next_fake = _dummy_transition(B)

        @nj.pure
        def run(trd, z_t, a_t, z_real, z_fake):
            scores_real = trd(z_t, a_t, z_real)
            scores_fake = trd(z_t, a_t, z_fake)
            loss_smooth = TRD.train_loss(scores_real, scores_fake, label_smooth=0.9)
            loss_hard = TRD.train_loss(scores_real, scores_fake, label_smooth=1.0)
            return loss_smooth, loss_hard

        state = {}
        state, (loss_s, loss_h) = run(
            state, trd, z_t, a_t, z_next_real, z_next_fake,
            seed=jax.random.PRNGKey(42), create=True)
        assert not jnp.allclose(loss_s, loss_h, atol=1e-3), \
            "Smoothed and hard losses should differ"

    def test_backward(self):
        """Gradients should flow through TRD."""
        B = 8
        trd = TRD(name='trd_test', hidden=64)
        z_t, a_t, z_next_real, z_next_fake = _dummy_transition(B)

        @nj.pure
        def run(trd, z_t, a_t, z_real, z_fake):
            scores_real = trd(z_t, a_t, z_real)
            scores_fake = trd(z_t, a_t, z_fake)
            return TRD.train_loss(scores_real, scores_fake)

        state = {}
        state, _ = run(state, trd, z_t, a_t, z_next_real, z_next_fake,
                        seed=jax.random.PRNGKey(42), create=True)

        def loss_fn(state):
            state, loss = run(state, trd, z_t, a_t, z_next_real, z_next_fake,
                              seed=jax.random.PRNGKey(42), create=True)
            return loss, state

        grads, _ = jax.grad(loss_fn, has_aux=True)(state)
        total_grad = sum(jax.tree.leaves(
            jax.tree.map(lambda x: jnp.abs(x).sum(), grads)))
        assert float(total_grad) > 0, "Gradients should be non-zero"


class TestSpectralNormLinear:

    def test_output_shape(self):
        """SpectralNormLinear should produce correct output shape."""
        B, in_dim, out_dim = 8, 64, 32
        layer = SpectralNormLinear(name='sn_test', units=out_dim)
        x = nn.cast(jax.random.normal(jax.random.PRNGKey(0), (B, in_dim)))

        @nj.pure
        def run(layer, x):
            return layer(x)

        state = {}
        state, out = run(state, layer, x,
                         seed=jax.random.PRNGKey(42), create=True)
        assert out.shape == (B, out_dim)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
