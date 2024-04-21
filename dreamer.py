def main():
  from sogym.env_gym import sogym
  import warnings
  import dreamerv3
  from dreamerv3 import embodied
  from gymnasium.wrappers import EnvCompatibility
  from gymnasium.wrappers import StepAPICompatibility

  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  # See configs.yaml for all options.
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['medium'])
  config = config.update({
      'logdir': './logdir/run1',
      'run.train_ratio': 64,
      'run.log_every': 30,  # Seconds
      'batch_size': 16,
      'encoder.mlp_keys': ['beta','volume','design_variables','n_steps_left'],
      'decoder.mlp_keys':  ['beta','volume','design_variables','n_steps_left'],
      'encoder.cnn_keys': ['image','structure_strain_energy'],
      'decoder.cnn_keys': ['image','structure_strain_energy'],
       'jax.platform': 'cpu',
      # 'encoder.resize':'stride3',
      # 'decoder.resize':'stride3',

  })
  config = embodied.Flags(config).parse()

  # Setup logging and agents
  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
  ])


  from embodied.envs import from_gym
  env = sogym(mode='train',observation_type='dense',vol_constraint_type = 'hard',resolution=50,img_format = 'HWC',check_connectivity=True) # Replace this with your Gym env.
  # env = StepAPICompatibility(env[0])
  env = from_gym.FromGym(env)

  env = dreamerv3.wrap_env(env, config)
  env = embodied.BatchEnv([env], parallel=False)

  agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
  replay = embodied.replay.Uniform(config.batch_length, config.replay_size, logdir / 'replay')
  args = embodied.Config(**config.run, logdir=config.logdir, batch_steps=config.batch_size * config.batch_length)
  embodied.run.train(agent, env, replay, logger, args)



if __name__ == '__main__':
  main()
