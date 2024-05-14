import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'   
import pathlib
import importlib
import sys
import warnings
from functools import partial as bind

directory = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(directory.parent))
sys.path.insert(0, str(directory.parent.parent))
__package__ = directory.name

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

import embodied
from embodied import wrappers
import dreamerv3


def main(argv=None):

  embodied.print(r"---  ___                           __   ______ ---")
  embodied.print(r"--- |   \ _ _ ___ __ _ _ __  ___ _ \ \ / /__ / ---")
  embodied.print(r"--- | |) | '_/ -_) _` | '  \/ -_) '/\ V / |_ \ ---")
  embodied.print(r"--- |___/|_| \___\__,_|_|_|_\___|_|  \_/ |___/ ---")

  from dreamerv3 import agent as agt
  
  parsed, other = embodied.Flags(configs=['defaults']).parse_known(argv)
  config = embodied.Config(agt.Agent.configs['defaults'])
  for name in parsed.configs:
    config = config.update(agt.Agent.configs[name])
  config = embodied.Flags(config).parse(other)
  config = config.update(
      logdir=config.logdir.format(timestamp=embodied.timestamp()))
  config = config.update({
      'run.train_ratio': 32,
        'enc.spaces': 'beta|design_variables|image|n_steps_left|score|structure_strain_energy|volume',
        'dec.spaces': 'beta|design_variables|image|n_steps_left|score|structure_strain_energy|volume'

  })
  args = embodied.Config(
      **config.run,
      logdir=config.logdir,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      batch_length_eval=config.batch_length_eval,
      replay_context=config.replay_context)
  print('Run script:', args.script)
  print('Logdir:', args.logdir)

  logdir = embodied.Path(args.logdir)
  if args.script not in ('env', 'replay'):
    logdir.mkdir()
    config.save(logdir / 'config.yaml')

  def init():
    embodied.timer.global_timer.enabled = args.timer
  embodied.distr.Process.initializers.append(init)
  init()

  if args.script == 'train':
    embodied.run.train(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_env, config),
        bind(make_logger, config), args)

  elif args.script == 'train_eval':
    embodied.run.train_eval(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, 'eval_replay', is_eval=True),
        bind(make_env, config),
        bind(make_eval_env, config),
        bind(make_logger, config), args)

  elif args.script == 'train_holdout':
    assert config.eval_dir
    embodied.run.train_holdout(
        bind(make_agent, config),
        bind(make_replay, config, 'replay'),
        bind(make_replay, config, config.eval_dir),
        bind(make_env, config),
        bind(make_logger, config), args)

  elif args.script == 'eval_only':
    embodied.run.eval_only(
        bind(make_agent, config),
        bind(make_env, config),
        bind(make_logger, config), args)

  elif args.script == 'parallel':
    embodied.run.parallel.combined(
        bind(make_agent, config),
        bind(make_replay, config, 'replay', rate_limit=True),
        bind(make_env, config),
        bind(make_logger, config), args)

  elif args.script == 'parallel_env':
    envid = args.env_replica
    if envid < 0:
      envid = int(os.environ['JOB_COMPLETION_INDEX'])
    embodied.run.parallel.env(
        bind(make_env, config), envid, args, False)

  elif args.script == 'parallel_replay':
      embodied.run.parallel.replay(
          bind(make_replay, config, 'replay', rate_limit=True), args)

  else:
    raise NotImplementedError(args.script)


def make_agent(config):
  from dreamerv3 import agent as agt
  env = make_env(config, 0)
  if config.random_agent:
    agent = embodied.RandomAgent(env.obs_space, env.act_space)
  else:
    agent = agt.Agent(env.obs_space, env.act_space, config)
  env.close()
  return agent


def make_logger(config):
  step = embodied.Counter()
  logdir = config.logdir
  multiplier = config.env.get(config.task.split('_')[0], {}).get('repeat', 1)
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(config.filter, 'Agent'),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.JSONLOutput(logdir, 'scores.jsonl', 'episode/score'),
      embodied.logger.TensorBoardOutput(
          logdir, config.run.log_video_fps, config.tensorboard_videos),
      # embodied.logger.WandbOutput(logdir.name, ...),
  ], multiplier)
  return logger


def make_replay(config, directory=None, is_eval=False, rate_limit=False):
  directory = directory and embodied.Path(config.logdir) / directory
  size = int(config.replay.size / 10 if is_eval else config.replay.size)
  if is_eval:
    size = 100
  length = config.batch_length
  kwargs = {}
  kwargs['online'] = config.replay.online
  if rate_limit and config.run.train_ratio > 0:
    kwargs['samples_per_insert'] = config.run.train_ratio / (
        length - config.replay_context)
    kwargs['tolerance'] = 5 * config.batch_size
    kwargs['min_size'] = min(
        max(config.batch_size, config.run.train_fill), size)
  selectors = embodied.replay.selectors
  if config.replay.fracs.uniform < 1 and not is_eval:
    assert config.jax.compute_dtype in ('bfloat16', 'float32'), (
        'Gradient scaling for low-precision training can produce invalid loss '
        'outputs that are incompatible with prioritized replay.')
    import numpy as np
    recency = 1.0 / np.arange(1, size + 1) ** config.replay.recexp
    kwargs['selector'] = selectors.Mixture(dict(
        uniform=selectors.Uniform(),
        priority=selectors.Prioritized(**config.replay.prio),
        recency=selectors.Recency(recency),
    ), config.replay.fracs)
  kwargs['chunksize'] = config.replay.chunksize
  replay = embodied.replay.Replay(length, size, directory, **kwargs)
  return replay


def make_env(config, env_id=0):
    from embodied.envs import from_gym
    from sogym.env_gym import sogym
    env = sogym(mode='train',observation_type='topopt_game',vol_constraint_type = 'hard',resolution=50,img_format = 'HWC',check_connectivity=True,use_std_strain=False) # Replace this with your Gym env.
    env = from_gym.FromGym(env)
    env = wrap_env(env, config)
    return env

def make_eval_env(config, env_id=0):
    from embodied.envs import from_gym
    from sogym.env_gym import sogym
    env = sogym(mode='test',observation_type='topopt_game',vol_constraint_type = 'hard',resolution=50,img_format = 'HWC',check_connectivity=True,use_std_strain=False) # Replace this with your Gym env.
    env = from_gym.FromGym(env)

    env = wrap_env(env, config)
    return env

def wrap_env(env, config):
  args = config.wrapper
  for name, space in env.act_space.items():
    if name == 'reset':
      continue
    elif not space.discrete:
      env = wrappers.NormalizeAction(env, name)
      if args.discretize:
        env = wrappers.DiscretizeAction(env, name, args.discretize)
  env = wrappers.ExpandScalars(env)
  if args.length:
    env = wrappers.TimeLimit(env, args.length, args.reset)
  if args.checks:
    env = wrappers.CheckSpaces(env)
  for name, space in env.act_space.items():
    if not space.discrete:
      env = wrappers.ClipAction(env, name)
  return env

if __name__ == '__main__':
  main()

  # python main.py --logdir ./logdir/test12m --configs size12m

  # python main.py   --logdir ./logdir/12m_initial --configs size12m --run.script train_eval --run.eval_eps 10 --replay.size 5e5 --run.num_envs 24

  #  python main.py   --logdir ./logdir/50m_initial --configs size50m --run.script train_eval --run.eval_eps 10 --replay.size 5e5 --run.num_envs 24

  # python main.py   --logdir ./logdir/200m_initial --configs size200m --run.script train_eval --run.eval_eps 10 --replay.size 5e5 --run.num_envs 24 --