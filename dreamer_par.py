def main():
  from sogym.env import sogym
  import warnings
  import dreamerv3
  from dreamerv3 import embodied
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['small'])
  config = config.update({
      'run.logdir': '~/logdir/run1',
      'run.train_ratio': 64,
      'run.log_every': 30,  # Seconds
      'batch_size': 16,
      'encoder.mlp_keys': ['conditions','volume','design_variables','n_steps_left'],
      'decoder.mlp_keys':  ['conditions','volume','design_variables','n_steps_left'],
      'encoder.cnn_keys': '$^',
      'decoder.cnn_keys': '$^',
       'jax.platform': 'cpu',
       'envs.amount':16,
  })

  logdir = embodied.Path(config.run.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
      # embodied.logger.WandBOutput(logdir.name, config),
      # embodied.logger.MLFlowOutput(logdir.name),
  ])

  import crafter
  from embodied.envs import from_gym

  import importlib
  import pathlib
  import sys
  import warnings
  from functools import partial as bind
  from embodied import wrappers

  env = sogym(nelx=100,nely=50,mode='train',observation_type='dense')  # Replace this with your Gym env.
  env = from_gym.FromGym(env)
  def make_envs(config, **overrides):
    suite, task = config.task.split('_', 1)
    ctors = []
    for index in range(config.envs.amount):
      ctor = lambda: wrap_env(env, config)
      if config.envs.parallel != 'none':
        ctor = bind(embodied.Parallel, ctor, config.envs.parallel)
      if config.envs.restart:
        ctor = bind(wrappers.RestartOnException, ctor)
      ctors.append(ctor)
    envs = [ctor() for ctor in ctors]
    return embodied.BatchEnv(envs, parallel=(config.envs.parallel != 'none'))
  def make_replay(
      config, directory=None, is_eval=False, rate_limit=False, **kwargs):
    assert config.replay == 'uniform' or not rate_limit
    length = config.batch_length
    size = config.replay_size // 10 if is_eval else config.replay_size
    if config.replay == 'uniform' or is_eval:
      kw = {'online': config.replay_online}
      if rate_limit and config.run.train_ratio > 0:
        kw['samples_per_insert'] = config.run.train_ratio / config.batch_length
        kw['tolerance'] = 10 * config.batch_size
        kw['min_size'] = config.batch_size
      replay = embodied.replay.Uniform(length, size, directory, **kw)
    elif config.replay == 'reverb':
      replay = embodied.replay.Reverb(length, size, directory)
    elif config.replay == 'chunks':
      replay = embodied.replay.NaiveChunks(length, size, directory)
    else:
      raise NotImplementedError(config.replay)
    return replay
  def wrap_env(env, config):
    args = config.wrapper
    for name, space in env.act_space.items():
      if name == 'reset':
        continue
      elif space.discrete:
        env = wrappers.OneHotAction(env, name)
      elif args.discretize:
        env = wrappers.DiscretizeAction(env, name, args.discretize)
      else:
        env = wrappers.NormalizeAction(env, name)
    env = wrappers.ExpandScalars(env)
    if args.length:
      env = wrappers.TimeLimit(env, args.length, args.reset)
    if args.checks:
      env = wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
      if not space.discrete:
        env = wrappers.ClipAction(env, name)
    return env
  args = config.run.update(batch_steps=config.batch_size * config.batch_length)

  replay = make_replay(config, logdir / 'replay')
  env = make_envs(config)
  agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
  embodied.run.train(agent, env, replay, logger, args)


if __name__ == '__main__':
  main()