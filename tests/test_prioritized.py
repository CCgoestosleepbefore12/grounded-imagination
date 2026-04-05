"""Unit tests for prioritized replay utilities."""

import jax.numpy as jnp
import pytest

import sys
sys.path.insert(0, '/Users/chengcheng/grounded_imagination')

from grounded.prioritized import (
    compute_priorities, importance_sampling_weights, beta_schedule
)


class TestComputePriorities:

    def test_low_trd_high_priority(self):
        """Low TRD score should give high priority."""
        trd_scores = jnp.array([0.1, 0.5, 0.9])
        prios = compute_priorities(trd_scores)
        # 0.1 TRD → ~0.9 priority, 0.9 TRD → ~0.1 priority
        assert float(prios[0]) > float(prios[1]) > float(prios[2])

    def test_output_positive(self):
        """Priorities should always be positive."""
        trd_scores = jnp.array([0.0, 0.5, 1.0])
        prios = compute_priorities(trd_scores)
        assert jnp.all(prios > 0)

    def test_shape(self):
        """Output shape should match input."""
        trd_scores = jnp.ones(16) * 0.5
        prios = compute_priorities(trd_scores)
        assert prios.shape == (16,)


class TestImportanceSamplingWeights:

    def test_shape(self):
        """IS weights should have same shape as priorities."""
        prios = jnp.array([0.1, 0.5, 0.9, 0.3])
        weights = importance_sampling_weights(prios, alpha=0.6, beta=0.4)
        assert weights.shape == (4,)

    def test_positive(self):
        """IS weights should be positive."""
        prios = jnp.array([0.1, 0.5, 0.9, 0.3])
        weights = importance_sampling_weights(prios, alpha=0.6, beta=0.4)
        assert jnp.all(weights > 0)

    def test_max_is_one(self):
        """Max IS weight should be ~1 after normalization."""
        prios = jnp.array([0.1, 0.5, 0.9, 0.3])
        weights = importance_sampling_weights(prios, alpha=0.6, beta=0.4)
        assert jnp.allclose(weights.max(), 1.0, atol=1e-5)

    def test_uniform_priorities(self):
        """When all priorities are equal, all IS weights should be equal."""
        prios = jnp.ones(8) * 0.5
        weights = importance_sampling_weights(prios, alpha=0.6, beta=0.4)
        assert jnp.allclose(weights, weights[0], atol=1e-5)

    def test_alpha_zero_uniform(self):
        """alpha=0 should give uniform sampling probabilities → equal weights."""
        prios = jnp.array([0.1, 0.5, 0.9, 0.3])
        weights = importance_sampling_weights(prios, alpha=0.0, beta=0.4)
        assert jnp.allclose(weights, weights[0], atol=1e-5)

    def test_high_priority_lower_weight(self):
        """High priority samples are oversampled → should get lower IS weight."""
        prios = jnp.array([0.9, 0.1])
        weights = importance_sampling_weights(prios, alpha=0.6, beta=1.0)
        # prios[0] is higher → sampled more → IS weight should be lower
        assert float(weights[0]) < float(weights[1])


class TestBetaSchedule:

    def test_start(self):
        """At step 0, beta should equal beta_start."""
        beta = beta_schedule(0, 1000, beta_start=0.4, beta_end=1.0)
        assert jnp.allclose(beta, 0.4, atol=1e-5)

    def test_end(self):
        """At final step, beta should equal beta_end."""
        beta = beta_schedule(1000, 1000, beta_start=0.4, beta_end=1.0)
        assert jnp.allclose(beta, 1.0, atol=1e-5)

    def test_midpoint(self):
        """At halfway, beta should be midpoint of start and end."""
        beta = beta_schedule(500, 1000, beta_start=0.4, beta_end=1.0)
        assert jnp.allclose(beta, 0.7, atol=1e-5)

    def test_clamp_beyond(self):
        """Beyond total_steps, beta should not exceed beta_end."""
        beta = beta_schedule(2000, 1000, beta_start=0.4, beta_end=1.0)
        assert jnp.allclose(beta, 1.0, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
