import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'   
import warnings
from functools import partial as bind

import dreamerv3
import embodied
from sogym.env_gym import sogym

warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')


def main():

  config = embodied.Config(dreamerv3.Agent.configs['defaults'])
  config = config.update({
      **dreamerv3.Agent.configs['size12m'],
      'logdir': f'./logdir/{embodied.timestamp()}-example',
      'run.train_ratio': 32,
      'run.num_envs': 10,
      'run.num_envs_eval': 1,
    #   'enc.simple.mlp_keys': ['beta','volume','design_variables','n_steps_left'],
    #   'dec.simple.mlp_keys':  ['beta','volume','design_variables','n_steps_left'],
    #   'enc.simple.cnn_keys': ['image','structure_strain_energy'],
    #   'dec.simple.cnn_keys': ['image','structure_strain_energy'],
    #   'jax.policy_devices': [1],
    #   'jax.train_devices' : [1],
    'enc.spaces':['image','structure_strain_energy'],
    'dec.spaces':['image','structure_strain_energy'],


  })
  config = embodied.Flags(config).parse()

  print('Logdir:', config.logdir)
  logdir = embodied.Path(config.logdir)
  logdir.mkdir()
  config.save(logdir / 'config.yaml')

  def make_agent(config):
    env = make_train_env(config)
    agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
    env.close()
    return agent

  def make_logger(config):
    logdir = embodied.Path(config.logdir)
    return embodied.Logger(embodied.Counter(), [
        embodied.logger.TerminalOutput(config.filter),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.TensorBoardOutput(logdir),
        # embodied.logger.WandbOutput(logdir.name, config=config),
    ])

  def make_replay(config, directory=None, is_eval=False, rate_limit=False):
    directory = directory and embodied.Path(config.logdir) / directory
    size = int(config.replay.size / 10 if is_eval else config.replay.size)
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
  def make_train_replay(config):
    return embodied.replay.Replay(
        length=config.batch_length,
        capacity=config.replay.size,
        directory=embodied.Path(config.logdir) / 'replay',
        online=config.replay.online)
  
  def make_eval_replay(config):
    return embodied.replay.Replay(
        length=config.batch_length,
        capacity=config.replay.size/10,
        directory=embodied.Path(config.logdir) / 'evalreplay',
        online=config.replay.online)


  def make_train_env(config, env_id=0):
    from embodied.envs import from_gym
    env = sogym(mode='train',observation_type='topopt_game',vol_constraint_type = 'hard',resolution=50,img_format = 'HWC',check_connectivity=True) # Replace this with your Gym env.
    # env = StepAPICompatibility(env[0])
    env = from_gym.FromGym(env)

    env = dreamerv3.wrap_env(env, config)
    return env

  def make_eval_env(config, env_id=0):
    from embodied.envs import from_gym
    env = sogym(mode='test',observation_type='topopt_game',vol_constraint_type = 'hard',resolution=50,img_format = 'HWC',check_connectivity=True) # Replace this with your Gym env.
    # env = StepAPICompatibility(env[0])
    env = from_gym.FromGym(env)

    env = dreamerv3.wrap_env(env, config)
    return env
  
  args = embodied.Config(
      **config.run,
      logdir=config.logdir,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      batch_length_eval=config.batch_length_eval,
      replay_context=config.replay_context,
  )

  embodied.run.train_eval(
      bind(make_agent, config),
      bind(make_replay, config, 'replay'),
      bind(make_replay, config, 'eval_replay', is_eval=True),
      bind(make_train_env, config),
      bind(make_eval_env,config),
      bind(make_logger, config), args)


if __name__ == '__main__':
  main()