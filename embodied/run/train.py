import collections
from functools import partial as bind

import elements
import embodied
import numpy as np


def train(make_agent, make_replay, make_env, make_stream, make_logger, args):

  agent = make_agent()
  replay = make_replay()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  train_agg = elements.Agg()
  epstats = elements.Agg()
  episodes = collections.defaultdict(elements.Agg)
  policy_fps = elements.FPS()
  train_fps = elements.FPS()

  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)
  should_save = embodied.LocalClock(args.save_every)

  # Grounded: DAgger correction environment
  correction_env = None
  dagger_counter = 0
  grounded = getattr(agent, 'grounded', False)
  if grounded:
    K_DAGGER = 128  # correct every 128 training steps
    correction_env = make_env(0)
    print('DAgger: created correction environment')

  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    for key, value in tran.items():
      if value.dtype == np.uint8 and value.ndim == 3:
        if worker == 0:
          episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
    if tran['is_last']:
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix='episode')
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(replay.add)
  driver.on_step(logfn)

  stream_train = iter(agent.stream(make_stream(replay, 'train')))
  stream_report = iter(agent.stream(make_stream(replay, 'report')))

  carry_train = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)

  # Store last batch for DAgger
  last_batch = [None]

  def trainfn(tran, worker):
    nonlocal dagger_counter
    if len(replay) < args.batch_size * args.batch_length:
      return
    for _ in range(should_train(step)):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      carry_train[0], outs, mets = agent.train(carry_train[0], batch)
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      train_agg.add(mets, prefix='train')

      # Grounded: DAgger correction
      if grounded and correction_env is not None:
        last_batch[0] = batch
        dagger_counter += 1
        if dagger_counter % K_DAGGER == 0 and 'trd_scores' in outs:
          _dagger_correct(outs, batch)

  def _dagger_correct(outs, batch):
    """Collect DAgger corrections at low-trust transitions."""
    trd_scores = np.asarray(outs['trd_scores'])  # (B, T-1)
    if 'qpos' not in batch or 'qvel' not in batch:
      return
    qpos = np.asarray(batch['qpos'])    # (B, T, nq)
    qvel = np.asarray(batch['qvel'])    # (B, T, nv)
    actions = {k: np.asarray(v) for k, v in batch.items()
               if k in agent.act_space}  # (B, T, act_dim)

    # Find worst transitions (lowest TRD scores)
    flat_scores = trd_scores.reshape(-1)
    n_correct = min(4, len(flat_scores))
    worst_idx = np.argpartition(flat_scores, n_correct)[:n_correct]

    corrections = 0
    for idx in worst_idx:
      b = idx // trd_scores.shape[1]
      t = idx % trd_scores.shape[1]
      t_actual = t + 1  # TRD scores are for transitions t→t+1, offset by 1

      if t_actual >= qpos.shape[1] - 1:
        continue

      try:
        # Set state and collect correct transition
        correction_env.set_state(qpos[b, t_actual], qvel[b, t_actual])
        act = {k: v[b, t_actual + 1] for k, v in actions.items()}
        act['reset'] = False
        obs = correction_env.step(act)

        # Add correction to replay with worker=-1 (special correction worker)
        replay.add(obs, worker=-1)
        corrections += 1
      except Exception as e:
        pass  # Skip failed corrections silently

    if corrections > 0:
      train_agg.add({'dagger/corrections': corrections}, prefix='train')

  driver.on_step(trainfn)

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  cp.replay = replay
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  cp.load_or_save()

  print('Start training loop')
  policy = lambda *args: agent.policy(*args, mode='train')
  driver.reset(agent.init_policy)
  while step < args.steps:

    driver(policy, steps=10)

    if should_report(step) and len(replay):
      agg = elements.Agg()
      for _ in range(args.consec_report * args.report_batches):
        carry_report, mets = agent.report(carry_report, next(stream_report))
        agg.add(mets)
      logger.add(agg.result(), prefix='report')

    if should_log(step):
      logger.add(train_agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

    if should_save(step):
      cp.save()

  if correction_env is not None:
    correction_env.close()
  logger.close()
