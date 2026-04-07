"""MetaWorld environment wrapper for DreamerV3."""

import functools

import elements
import embodied
import numpy as np


class MetaWorld(embodied.Env):

  def __init__(self, task, size=(64, 64), camera='corner2', seed=0, **kwargs):
    import metaworld
    # MetaWorld task names use '-' (e.g., 'pick-place-v2')
    task_name = task.replace('_', '-')
    if not task_name.endswith(('-v2', '-v3')):
      task_name = task_name + '-v3'
    ml1 = metaworld.ML1(task_name, seed=seed)
    env = ml1.train_classes[task_name]()
    env.set_task(ml1.train_tasks[0])
    self._env = env
    self._size = size
    self._camera = camera
    self._done = True
    self._info = None
    self._seed = seed

  @functools.cached_property
  def obs_space(self):
    obs = self._env.observation_space
    return {
        'image': elements.Space(np.uint8, (*self._size, 3), 0, 255),
        'proprio': elements.Space(np.float32, obs.shape, obs.low, obs.high),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @functools.cached_property
  def act_space(self):
    act = self._env.action_space
    return {
        'action': elements.Space(np.float32, act.shape, act.low, act.high),
        'reset': elements.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      obs, self._info = self._env.reset()
      return self._obs(obs, 0.0, is_first=True)
    obs, reward, terminated, truncated, self._info = self._env.step(
        action['action'])
    self._done = terminated or truncated
    return self._obs(
        obs, reward,
        is_last=self._done,
        is_terminal=terminated)

  def _obs(self, obs, reward, is_first=False, is_last=False, is_terminal=False):
    image = self._render()
    return {
        'image': image,
        'proprio': np.float32(obs),
        'reward': np.float32(reward),
        'is_first': is_first,
        'is_last': is_last,
        'is_terminal': is_terminal,
    }

  def _render(self):
    import mujoco
    # Use MuJoCo offscreen rendering directly
    model = self._env.model
    data = self._env.data
    renderer = mujoco.Renderer(model, *self._size)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    image = renderer.render()
    renderer.close()
    return np.uint8(image)

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass
