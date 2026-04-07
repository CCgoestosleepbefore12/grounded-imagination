"""MetaWorld environment wrapper for DreamerV3."""

import functools

import elements
import embodied
import numpy as np


class MetaWorld(embodied.Env):

  def __init__(self, task, size=(64, 64), camera='corner2', seed=0,
               image=True, **kwargs):
    import metaworld
    task_name = task.replace('_', '-')
    if not task_name.endswith(('-v2', '-v3')):
      task_name = task_name + '-v3'
    ml1 = metaworld.ML1(task_name, seed=seed)
    env = ml1.train_classes[task_name]()
    env.set_task(ml1.train_tasks[0])
    self._env = env
    self._size = size
    self._camera = camera
    self._image = image
    self._done = True
    self._info = None
    self._renderer = None
    self._nq = env.model.nq
    self._nv = env.model.nv

  @functools.cached_property
  def obs_space(self):
    obs = self._env.observation_space
    spaces = {
        'proprio': elements.Space(np.float32, obs.shape, obs.low, obs.high),
        'qpos': elements.Space(np.float32, (self._nq,)),
        'qvel': elements.Space(np.float32, (self._nv,)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }
    if self._image:
      spaces['image'] = elements.Space(np.uint8, (*self._size, 3), 0, 255)
    return spaces

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
    result = {
        'proprio': np.float32(obs),
        'qpos': np.float32(self._env.data.qpos.copy()),
        'qvel': np.float32(self._env.data.qvel.copy()),
        'reward': np.float32(reward),
        'is_first': is_first,
        'is_last': is_last,
        'is_terminal': is_terminal,
    }
    if self._image:
      result['image'] = self._render()
    return result

  def set_state(self, qpos, qvel):
    """Set MuJoCo state for DAgger action replay."""
    import mujoco
    self._env.data.qpos[:] = qpos
    self._env.data.qvel[:] = qvel
    mujoco.mj_forward(self._env.model, self._env.data)
    self._done = False  # prevent step() from triggering reset

  def _render(self):
    import mujoco
    if self._renderer is None:
      self._renderer = mujoco.Renderer(self._env.model, *self._size)
    self._renderer.update_scene(self._env.data)
    return np.uint8(self._renderer.render())

  def close(self):
    try:
      if self._renderer is not None:
        self._renderer.close()
        self._renderer = None
      self._env.close()
    except Exception:
      pass
