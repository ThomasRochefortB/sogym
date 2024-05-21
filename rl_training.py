import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'       #Disactivate multiprocessing for numpy
import comet_ml
import numpy as np
import gymnasium as gym
import yaml
from datetime import datetime

from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, VecCheckNan
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, CheckpointCallback, StopTrainingOnNoModelImprovement

from sogym.env import sogym
from sogym.utils import ImageDictExtractor, CustomBoxDense
from sogym.callbacks import FigureRecorderCallback, MaxRewardCallback, GradientNormCallback, GradientClippingCallback
import os
import torch
import multiprocessing
import argparse

import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

# Initialize parser
parser = argparse.ArgumentParser(description='Pretrain RL agent with expert dataset.')

parser.add_argument('--observation-type', type=str, default="topopt_game",
                    help='Type of observation used in the environment.')

parser.add_argument('--vol-constraint-type', type=str, default="hard")

parser.add_argument('--use-std-strain', action='store_true', help='Use standard strain in the observation.')


parser.add_argument('--algorithm-name', type=str, default="PPO",
                    help='Algorithm to use for training (SAC, PPO, TD3).')

parser.add_argument('--algorithm-config-file', type=str, default="algorithms.yaml",
                    help='Path to the algorithm configuration file.')

parser.add_argument('--resume', action='store_true', help='Resume training from a saved model.')

parser.add_argument('--resume-path', type=str, default="")

parser.add_argument('--training-phase', type=str, default='naive')

parser.add_argument('--replay-buffer-path', type=str, default='')

parser.add_argument('--restart-run', type=str, default='')

parser.add_argument('--log-comet', action='store_true', help='Log to comet.ml.')

args = parser.parse_args() 
def main():


    if args.log_comet:
        comet_ml.init(project_name="rl_training")
        experiment = comet_ml.Experiment(api_key="No20MKxPKu7vWLOUQCFBRO8mo")

    # Set number of CPUs to use automatically
    num_cpu = multiprocessing.cpu_count()*2
    print(f"Using {num_cpu} CPUs!")

    algorithm_name = args.algorithm_name  # or "TD3"

    # Load the YAML file and extract parameters
    with open(args.algorithm_config_file, "r") as file:
        config = yaml.safe_load(file)

    common_params = config.get("common", {})
    algorithm_params = config.get(algorithm_name, {})

    # Define other parameters
    params = {
        'observation_type': args.observation_type,
        'vol_constraint_type': args.vol_constraint_type,
        'use_std_strain': args.use_std_strain,
        'check_connectivity': True,
        'resolution': 50
    }

    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_critic_only = False  # if True, we freeze everything except the critic
    pretrained_run = None
    restart_run = args.restart_run  # or "PPO_20240516_093938"

    log_name = restart_run if restart_run else f"{algorithm_name}_{current_datetime}_{os.getpid()}"

    if args.log_comet:
        experiment.set_name(log_name)
    # Create directory and config file if needed
    log_dir = f'./runs/{log_name}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        # Combine all parameters for saving to the config file
        save_params = {**algorithm_params, **params, 'algorithm_name': algorithm_name, **common_params}
        with open(f'{log_dir}/config.yaml', 'w') as file:
            yaml.dump(save_params, file)

    # Create training and evaluation environments
    train_env = sogym(mode='train', **params)
    env = make_vec_env(lambda: train_env, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)
    env = VecCheckNan(env, raise_exception=True)

    eval_env = sogym(mode='test', observation_type = args.observation_type, vol_constraint_type = 'hard', use_std_strain = False, check_connectivity = True, resolution = 50)
    eval_env = make_vec_env(lambda: eval_env, n_envs=1, vec_env_cls=SubprocVecEnv)
        # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))

    chosen_policy = "MlpPolicy" if args.observation_type == 'vector' else "MultiInputPolicy"
    feature_extractor = ImageDictExtractor if args.observation_type == 'image' or args.observation_type=="topopt_game" else CustomBoxDense


    policy_kwargs = dict(
        features_extractor_class=feature_extractor,
        net_arch = config['common']['net_arch'],
        share_features_extractor = False,
    )

    # Create the model based on the algorithm name and parameters
    if algorithm_name == "SAC":
        model = SAC(env=env,
                    policy = chosen_policy, 
                    policy_kwargs=policy_kwargs,
                    #action_noise = action_noise,
                    ent_coef = 0.0,
                    device=device, 
                    **algorithm_params)

    elif algorithm_name == "PPO":
        model = PPO(env=env, 
                    policy = chosen_policy, 
                    policy_kwargs=policy_kwargs,
                    n_steps= 64*386 // num_cpu,
                    batch_size= 16384//4,
                    tensorboard_log  ='./runs/{}'.format(log_name),
                    device = device, 
                    **algorithm_params)

    elif algorithm_name == "TD3":
        # Create the action noise object
        n_actions = env.action_space.shape[-1]
        action_noise_params = algorithm_params.pop("action_noise")
        action_noise = NormalActionNoise(mean=action_noise_params["mean"] * np.ones(n_actions),
                                        sigma=action_noise_params["sigma"] * np.ones(n_actions))
        model = TD3(env=env,
                    policy =chosen_policy, 
                    policy_kwargs=policy_kwargs,
                    action_noise=action_noise,
                    device=device, 
                    **algorithm_params)

    if pretrained_run:
        model.set_parameters("./runs/{}/checkpoints/{}".format(log_name,pretrained_run))

    if restart_run:
        model = model.load("./runs/{}/checkpoints/best_model.zip".format(log_name),env=env)
                    
    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
    save_freq=250_000//num_cpu,
    save_path="./runs/{}/checkpoints/".format(log_name),
    name_prefix=log_name,
    save_replay_buffer=True,
    save_vecnormalize=True,
    )

    eval_callback = EvalCallback(eval_env,
                                log_path='./runs/{}/'.format(log_name), 
                                eval_freq=10_000//num_cpu,
                                deterministic=True,
                                n_eval_episodes=10,
                                render=False,
                                best_model_save_path='./runs/{}/checkpoints/'.format(log_name),
                                verbose=0)

    callback_list = CallbackList([eval_callback,
                            checkpoint_callback,
                            MaxRewardCallback(verbose=1),
                            GradientClippingCallback(clip_value=1.0, verbose=1),
                            GradientNormCallback(verbose=1),
                            FigureRecorderCallback(eval_env=eval_env, check_freq=10_000//num_cpu, figure_size=(8, 6))
                            ])
    

    if train_critic_only:
        #Freeze everything:
        for name, param in model.policy.named_parameters():
            if param.requires_grad:
                param.requires_grad=False

        if algorithm_name =='SAC':
            # Unfreeze critic:
            for param in model.policy.critic.parameters():
                if param.requires_grad==False:
                    param.requires_grad=True

            for param in model.policy.critic_target.parameters():
                if param.requires_grad==False:
                    param.requires_grad=True


        if algorithm_name == 'PPO':
            for param in model.policy.mlp_extractor.value_net.parameters():
                if param.requires_grad==False:
                    param.requires_grad=True
                
            for param in model.policy.value_net.parameters():
                if param.requires_grad==False:
                    param.requires_grad=True
    
    model.learn(25_000_000,
            callback=callback_list, 
            tb_log_name=log_name,
            reset_num_timesteps=not restart_run
            )

    # save the model:
    model.save('./runs/{}/checkpoints/final_model')
    if algorithm_name != 'PPO':
        model.save_replay_buffer("./runs/{}/checkpoints/final_buffer")

if __name__ == "__main__":
    main()


##!python rl_training.py --observation-type topopt_game --algorithm-name SAC --training-phase naive