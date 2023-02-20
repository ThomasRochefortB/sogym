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
      'encoder.cnn_keys': 'image',
      'decoder.cnn_keys': 'image',
       'jax.platform': 'cpu',
       'encoder.resize':'stride3',
       'decoder.resize':'stride3',

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
  env = sogym(nelx=100,nely=50,mode='train',observation_type='image')  # Replace this with your Gym env.
  env = from_gym.FromGym(env)
  env = dreamerv3.wrap_env(env, config)
  env = embodied.BatchEnv([env], parallel=False)

  agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
  replay = embodied.replay.Uniform(
      config.batch_length, config.replay_size, logdir / 'replay')
  args = config.run.update(batch_steps=config.batch_size * config.batch_length)
  embodied.run.train(agent, env, replay, logger, args)


if __name__ == '__main__':
  main()