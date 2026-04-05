"""Unit tests for diagnostic tools."""

import jax
import jax.numpy as jnp
import numpy as np
import ninjax as nj
import pytest
import embodied.jax.nets as nn

import sys
sys.path.insert(0, '/Users/chengcheng/grounded_imagination')

from grounded.diagnostics import (
    trd_scores_over_episode,
    router_entropy,
    effective_imagination_steps,
    training_metrics_summary,
)
from grounded.trd import TRD


class TestTrdScoresOverEpisode:

    def test_shapes(self):
        """Output shapes should be (T-1,)."""
        T, z_dim, a_dim = 20, 64, 4
        trd = TRD(name='trd_diag', hidden=32)
        feat = nn.cast(jax.random.normal(jax.random.PRNGKey(0), (T, z_dim)))
        actions = nn.cast(jax.random.normal(jax.random.PRNGKey(1), (T, a_dim)))

        @nj.pure
        def run(trd, feat, actions):
            return trd_scores_over_episode(trd, feat, actions)

        state = {}
        state, (per_step, cumulative) = run(
            state, trd, feat, actions,
            seed=jax.random.PRNGKey(42), create=True)
        assert per_step.shape == (T - 1,)
        assert cumulative.shape == (T - 1,)

    def test_values_in_range(self):
        """Trust scores should be in (0, 1)."""
        T, z_dim, a_dim = 10, 32, 2
        trd = TRD(name='trd_diag', hidden=16)
        feat = nn.cast(jax.random.normal(jax.random.PRNGKey(0), (T, z_dim)))
        actions = nn.cast(jax.random.normal(jax.random.PRNGKey(1), (T, a_dim)))

        @nj.pure
        def run(trd, feat, actions):
            return trd_scores_over_episode(trd, feat, actions)

        state = {}
        state, (per_step, cumulative) = run(
            state, trd, feat, actions,
            seed=jax.random.PRNGKey(42), create=True)
        assert jnp.all(per_step > 0) and jnp.all(per_step < 1)
        assert jnp.all(cumulative > 0) and jnp.all(cumulative <= 1)

    def test_cumulative_monotone(self):
        """Cumulative trust should be monotonically non-increasing."""
        T, z_dim, a_dim = 15, 32, 2
        trd = TRD(name='trd_diag', hidden=16)
        feat = nn.cast(jax.random.normal(jax.random.PRNGKey(0), (T, z_dim)))
        actions = nn.cast(jax.random.normal(jax.random.PRNGKey(1), (T, a_dim)))

        @nj.pure
        def run(trd, feat, actions):
            return trd_scores_over_episode(trd, feat, actions)

        state = {}
        state, (_, cumulative) = run(
            state, trd, feat, actions,
            seed=jax.random.PRNGKey(42), create=True)
        diffs = cumulative[1:] - cumulative[:-1]
        assert jnp.all(diffs <= 1e-5), "Cumulative trust should be non-increasing"


class TestRouterEntropy:

    def test_uniform_max_entropy(self):
        """Uniform weights should give maximum entropy = log(K)."""
        K = 4
        weights = jnp.ones((8, K)) / K
        ent = router_entropy(weights)
        expected = jnp.log(K)
        assert jnp.allclose(ent, expected, atol=1e-5)

    def test_peaked_low_entropy(self):
        """One-hot weights should give near-zero entropy."""
        weights = jnp.array([[1.0, 0.0, 0.0, 0.0]] * 8)
        ent = router_entropy(weights)
        assert float(ent) < 0.01

    def test_intermediate(self):
        """Non-uniform weights should give entropy between 0 and log(K)."""
        weights = jnp.array([[0.7, 0.1, 0.1, 0.1]] * 8)
        ent = router_entropy(weights)
        assert 0 < float(ent) < float(jnp.log(4))


class TestEffectiveImaginationSteps:

    def test_all_above(self):
        """All steps above tau → effective steps = H."""
        cum_trust = jnp.ones((4, 10)) * 0.5
        steps = effective_imagination_steps(cum_trust, tau=0.15)
        assert jnp.allclose(steps, 10.0)

    def test_all_below(self):
        """All steps below tau → effective steps = 0."""
        cum_trust = jnp.ones((4, 10)) * 0.01
        steps = effective_imagination_steps(cum_trust, tau=0.15)
        assert jnp.allclose(steps, 0.0)

    def test_partial(self):
        """Some steps above, some below → correct count."""
        cum_trust = jnp.array([
            [0.9, 0.8, 0.5, 0.2, 0.1],  # 4 above 0.15 (0.9,0.8,0.5,0.2)
            [0.9, 0.1, 0.05, 0.01, 0.0],  # 1 above 0.15 (0.9)
        ])
        steps = effective_imagination_steps(cum_trust, tau=0.15)
        expected = (4 + 1) / 2.0
        assert jnp.allclose(steps, expected)


class TestTrainingMetricsSummary:

    def test_basic(self):
        """Should aggregate metrics into arrays."""
        history = [
            {'trd_score_real': 0.5, 'return': 10.0},
            {'trd_score_real': 0.6, 'return': 20.0},
            {'trd_score_real': 0.7, 'return': 30.0},
        ]
        summary = training_metrics_summary(history)
        assert 'trd_score_real' in summary
        assert len(summary['trd_score_real']) == 3
        np.testing.assert_allclose(summary['return'], [10, 20, 30])

    def test_empty(self):
        """Empty history should return empty dict."""
        assert training_metrics_summary([]) == {}

    def test_missing_keys(self):
        """Should handle metrics that appear in some steps but not others."""
        history = [
            {'a': 1.0, 'b': 2.0},
            {'a': 3.0},
            {'a': 5.0, 'b': 6.0},
        ]
        summary = training_metrics_summary(history)
        assert len(summary['a']) == 3
        assert len(summary['b']) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
