"""Unit tests for DAgger correction module."""

import jax.numpy as jnp
import numpy as np
import pytest

import sys
sys.path.insert(0, '/Users/chengcheng/grounded_imagination')

from grounded.dagger import find_trust_boundary, action_replay_correct, collect_corrections


class DummyEnv:
    """Minimal mock environment for testing action replay."""

    def __init__(self, qpos_dim=3, qvel_dim=3, obs_dim=6, act_dim=2):
        self.qpos = np.zeros(qpos_dim)
        self.qvel = np.zeros(qvel_dim)
        self.obs_dim = obs_dim
        self.step_count = 0

    def set_state(self, qpos, qvel):
        self.qpos = np.array(qpos, dtype=np.float64)
        self.qvel = np.array(qvel, dtype=np.float64)
        self.step_count = 0

    def get_state(self):
        return self.qpos.copy(), self.qvel.copy()

    def get_obs(self):
        return np.concatenate([self.qpos, self.qvel])

    def step(self, action):
        action = np.asarray(action)
        self.qpos = self.qpos + action[:len(self.qpos)] * 0.1
        self.qvel = self.qvel + action[:len(self.qvel)] * 0.05
        self.step_count += 1
        obs = self.get_obs()
        reward = -float(np.sum(self.qpos ** 2))
        done = False
        info = {}
        return obs, reward, done, info


class TestFindTrustBoundary:

    def test_basic_drop(self):
        """Should find the first step below tau."""
        cum_trust = jnp.array([
            [0.9, 0.8, 0.7, 0.3, 0.1],  # drops below 0.5 at index 3
            [0.9, 0.8, 0.6, 0.5, 0.4],  # drops below 0.5 at index 4
        ])
        result = find_trust_boundary(cum_trust, tau=0.5)
        expected = jnp.array([3, 4])
        assert jnp.array_equal(result, expected), f"Got {result}"

    def test_no_drop(self):
        """Should return H when trust never drops below tau."""
        cum_trust = jnp.array([
            [0.9, 0.8, 0.7, 0.6, 0.55],
        ])
        result = find_trust_boundary(cum_trust, tau=0.5)
        assert int(result[0]) == 5, f"Expected 5 (H), got {int(result[0])}"

    def test_immediate_drop(self):
        """Should return 0 when first step is already below tau."""
        cum_trust = jnp.array([
            [0.1, 0.05, 0.01],
        ])
        result = find_trust_boundary(cum_trust, tau=0.5)
        assert int(result[0]) == 0

    def test_batch(self):
        """Should handle batched inputs correctly."""
        cum_trust = jnp.array([
            [0.9, 0.8, 0.7, 0.6, 0.5],  # no drop (all >= 0.15)
            [0.9, 0.8, 0.1, 0.05, 0.01],  # drops at index 2
            [0.1, 0.05, 0.01, 0.0, 0.0],  # drops at index 0
        ])
        result = find_trust_boundary(cum_trust, tau=0.15)
        expected = jnp.array([5, 2, 0])
        assert jnp.array_equal(result, expected), f"Got {result}"


class TestActionReplayCorrect:

    def test_basic_replay(self):
        """Should replay actions and return correction data."""
        env = DummyEnv(qpos_dim=2, qvel_dim=2, act_dim=2)
        qpos_0 = np.array([1.0, 2.0])
        qvel_0 = np.array([0.0, 0.0])
        actions = np.array([
            [0.5, 0.5],
            [1.0, 1.0],
            [0.2, 0.3],
            [0.1, 0.1],
        ])
        result = action_replay_correct(env, qpos_0, qvel_0, actions, boundary_k=2)

        assert result is not None
        assert result['action'].shape == (2,)
        assert np.allclose(result['action'], [0.2, 0.3])
        assert result['reward'] is not None
        assert result['next_obs'] is not None
        # Env should have stepped 3 times (2 replay + 1 correction)
        assert env.step_count == 3

    def test_boundary_at_zero(self):
        """Should correct at step 0 without any replay."""
        env = DummyEnv(qpos_dim=2, qvel_dim=2, act_dim=2)
        qpos_0 = np.array([0.0, 0.0])
        qvel_0 = np.array([0.0, 0.0])
        actions = np.array([[1.0, 1.0], [2.0, 2.0]])
        result = action_replay_correct(env, qpos_0, qvel_0, actions, boundary_k=0)

        assert result is not None
        assert np.allclose(result['action'], [1.0, 1.0])
        assert env.step_count == 1

    def test_boundary_beyond_length(self):
        """Should return None if boundary_k >= H."""
        env = DummyEnv()
        actions = np.array([[1.0, 1.0], [2.0, 2.0]])
        result = action_replay_correct(
            env, np.zeros(3), np.zeros(3), actions, boundary_k=5)
        assert result is None


class TestCollectCorrections:

    def test_collects_at_boundary(self):
        """Should collect correction at trust boundary."""
        env = DummyEnv(qpos_dim=2, qvel_dim=2, act_dim=2)
        qpos_0 = np.array([0.0, 0.0])
        qvel_0 = np.array([0.0, 0.0])
        actions = np.array([
            [0.5, 0.5], [1.0, 1.0], [0.2, 0.3], [0.1, 0.1],
        ])
        cum_trust = np.array([0.9, 0.8, 0.3, 0.1])  # drops below 0.5 at idx 2

        corrections = collect_corrections(
            env, qpos_0, qvel_0, actions, cum_trust, tau=0.5, max_corrections=1)
        assert len(corrections) == 1
        assert np.allclose(corrections[0]['action'], [0.2, 0.3])

    def test_no_correction_needed(self):
        """Should return empty list when trust is always high."""
        env = DummyEnv(qpos_dim=2, qvel_dim=2, act_dim=2)
        actions = np.array([[0.5, 0.5], [1.0, 1.0]])
        cum_trust = np.array([0.9, 0.8])

        corrections = collect_corrections(
            env, np.zeros(2), np.zeros(2), actions, cum_trust, tau=0.5)
        assert len(corrections) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
