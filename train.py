def main():
    from sogym.env import sogym
    from sogym.utils import FigureRecorderCallback
    import numpy as np
    import stable_baselines3
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, CallbackList
    from sogym.utils import ImageDictExtractor
    import wandb
    from wandb.integration.sb3 import WandbCallback
    import argparse

    # Let's parse some arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_timesteps', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--obs_type', type=str, default='dense')
    parser.add_argument('--n_envs', type=int, default=1)
    parser.add_argument('--normalize', type=bool, default=True)
    parser.add_argument('--l2_reg', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_steps', type=int, default=2048)
    parser.add_argument('--feature_extractor', type=str, default='default')
    parser.add_argument('--net_arch', type=str, default='default')
    args = parser.parse_args()

    print('SB3 version:', stable_baselines3.__version__)
    # Let's make the code device agnostic:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    num_cpu = args.n_envs # Number of processes to use
    print(num_cpu)
    train_env = sogym(mode='train',observation_type=args.obs_type)
    env= make_vec_env(lambda:train_env, n_envs=num_cpu,vec_env_cls=SubprocVecEnv)
    if args.normalize:
        env=VecNormalize(env,gamma=1.0)

    eval_env = sogym(mode='test',observation_type=args.obs_type)
    eval_env = make_vec_env(lambda:eval_env, n_envs=1,vec_env_cls=SubprocVecEnv)
    if args.normalize:
        eval_env =VecNormalize(eval_env,gamma=1.0)

    config = {
        "policy_type": "MultiInputPolicy",
        "total_timesteps": args.total_timesteps,
    }
    run = wandb.init(
        project="sogym_trial_A",
        #name=,
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # monitor_gym=True,  # auto-upload the videos of agents playing the game
        # save_code=True,  # optional
    )

    eval_callback = EvalCallback(eval_env,log_path='tb_logs',eval_freq=100,deterministic=True,render=False,verbose=0)
    callback = CallbackList([eval_callback,
                            FigureRecorderCallback(check_freq=2*8*32),
                            WandbCallback(
            model_save_path=f"models/{run.id}",
            gradient_save_freq=5000,
            model_save_freq=0,
            verbose=0,
        ),])


    policy_kwargs=dict()
    if args.feature_extractor != 'default':
        policy_kwargs['features_extractor_class']=ImageDictExtractor
    if args.net_arch != 'default':
        policy_kwargs['net_arch']=dict(pi=[512,512,512], vf=[512,512,512])
    if args.l2_reg != 0.0:
        policy_kwargs['optimizer_kwargs']={"weight_decay": args.l2_reg}

    model = PPO(config['policy_type'],
                env,
                seed=args.seed,
                policy_kwargs = policy_kwargs,
                n_steps=args.n_steps//num_cpu,
                batch_size=args.batch_size,
                verbose=0,
                n_epochs=3,
                vf_coef = 1.0,
                clip_range = 0.3,
                clip_range_vf = 10.0,
                target_kl = 0.02,
                gamma=1.0, 
                learning_rate=3e-4,
                ent_coef=3e-4,
                tensorboard_log="tb_logs",
                device=device
                )

    model.learn(config['total_timesteps'],
                callback=callback,
    )
    run.finish()

    model.save('model_saved',)
    env.save('env_saved.pkl')


if __name__ == '__main__':
  main()