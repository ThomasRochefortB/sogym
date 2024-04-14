import os
import comet_ml
import numpy as np
import gymnasium as gym
import yaml
from datetime import datetime

from stable_baselines3.common.noise import NormalActionNoise
import stable_baselines3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
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
parser.add_argument('--algorithm-name', type=str, default="SAC",
                    help='Algorithm to use for training (SAC, PPO, TD3).')
parser.add_argument('--algorithm-config-file', type=str, default="algorithms.yaml",
                    help='Path to the algorithm configuration file.')

parser.add_argument('--resume', action='store_true', help='Resume training from a saved model.')

parser.add_argument('--resumepath', type=str, default="")

parser.add_argument('--training-phase', type=str, default='naive')


args = parser.parse_args()
def main():


    comet_ml.init(project_name="rl_training")
    experiment = comet_ml.Experiment(api_key="No20MKxPKu7vWLOUQCFBRO8mo")

    # Set up the environment
    observation_type = "topopt_game"
    observation_type = args.observation_type
    algorithm_name = args.algorithm_name  
    algorithm_config_file = args.algorithm_config_file

    chosen_policy = "MlpPolicy" if observation_type == 'box_dense' else "MultiInputPolicy"
    feature_extractor = ImageDictExtractor if observation_type == 'image' or observation_type == 'topopt_game' else CustomBoxDense
 

    # Set up multiprocessing
    num_cpu = multiprocessing.cpu_count()
    train_env = make_vec_env(lambda: sogym(mode='train', observation_type=observation_type, vol_constraint_type='hard', resolution=50, check_connectivity=True), n_envs=num_cpu, vec_env_cls=SubprocVecEnv)
    eval_env = make_vec_env(lambda: sogym(mode='test', observation_type=observation_type, vol_constraint_type='hard', resolution=50, check_connectivity=False), n_envs=1, vec_env_cls=SubprocVecEnv)


    # Set up the model
    with open(algorithm_config_file, "r") as file:
        config = yaml.safe_load(file)

    algorithm_params = config[algorithm_name]

    policy_kwargs = dict(
        features_extractor_class=feature_extractor,
        net_arch=config['common']['net_arch'],
        share_features_extractor=False
    )

    # Create the model based on the algorithm name and parameters
    if algorithm_name == "SAC":
        model = SAC(env=train_env,
                    policy = chosen_policy, 
                    policy_kwargs=policy_kwargs,
                    device=device, 
                    **algorithm_params)

    elif algorithm_name == "PPO":
        model = PPO(env=train_env, 
                    policy = chosen_policy, 
                    policy_kwargs=policy_kwargs,
                    device = device, 
                    **algorithm_params)

    elif algorithm_name == "TD3":
        # Create the action noise object
        n_actions = train_env.action_space.shape[-1]
        action_noise_params = algorithm_params.pop("action_noise")
        action_noise = NormalActionNoise(mean=action_noise_params["mean"] * np.ones(n_actions),
                                        sigma=action_noise_params["sigma"] * np.ones(n_actions))
        model = TD3(env=train_env,
                    policy =chosen_policy, 
                    policy_kwargs=policy_kwargs,
                    action_noise=action_noise,
                    device=device, 
                    **algorithm_params)
        
    # Set up callbacks
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_log_name = f"{algorithm_name}_{current_datetime}"

    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // num_cpu,
        save_path="./checkpoints/",
        name_prefix=tb_log_name,
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    early_stopping = StopTrainingOnNoModelImprovement(max_no_improvement_evals=50, min_evals=100, verbose=1)

    eval_callback = EvalCallback(eval_env,
                                 log_path='tb_logs',
                                 eval_freq=5000 // num_cpu,
                                 deterministic=True,
                                 n_eval_episodes=10,
                                 render=False,
                                 best_model_save_path='./checkpoints',
                                 callback_after_eval=early_stopping,
                                 verbose=0)


    callback_list = CallbackList([eval_callback,
                                  checkpoint_callback,
                                  MaxRewardCallback(verbose=1),
                                  GradientClippingCallback(clip_value=10.0, verbose=1),
                                  GradientNormCallback(verbose=1),
                                  FigureRecorderCallback(check_freq=5000 // num_cpu, eval_env=eval_env),
                                  ])
    
    if args.training_phase =='critic' and not args.resume:
        model.set_parameters(args.imitation_model_path)
    if args.training_phase =='actor-critic' and not args.resume:
        model.set_parameters(args.critic_model_path)

    if args.training_phase =='critic' and args.resume:
        model.set_parameters(args.resume_path)
    if args.training_phase =='actor-critic' and args.resume:
        model.set_parameters(args.resume_path)

    if args.training_phase =='naive'and args.resume:
        model.set_parameters(args.resume_path)

    if args.training_phase =='critic':
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
            
            #Reset critic networks:
            if hasattr(model.policy.critic_target, 'reset_parameters'):
                print(' resetting')
                model.policy.critic_target.reset_parameters()
            else:
                model.policy.critic_target.apply(init_weights)

                
            if hasattr(model.policy.critic, 'reset_parameters'):
                print('resetting')
                model.policy.critic_target.reset_parameters() 
            else:
                model.policy.critic.apply(init_weights)


        if algorithm_name == 'PPO':
            for param in model.policy.mlp_extractor.value_net.parameters():
                if param.requires_grad==False:
                    param.requires_grad=True
                
            for param in model.policy.value_net.parameters():
                if param.requires_grad==False:
                    param.requires_grad=True
            

            if hasattr(model.policy.value_net, 'reset_parameters'):
                print(' resetting')
                model.policy.value_net.reset_parameters()
            else:
                model.policy.value_net.apply(init_weights)
                
            if hasattr(model.policy.mlp_extractor.value_net, 'reset_parameters'):
                print(' resetting')
                model.policy.mlp_extractor.value_net.reset_parameters() 
            else:
                model.policy.mlp_extractor.value_net.apply(init_weights)


    # Train the model
    model.learn(20000000,
                callback=callback_list,
                tb_log_name=tb_log_name)

    # Save the model
    model.save('checkpoints/{}_final'.format(tb_log_name))

if __name__ == "__main__":
    main()


##!python rl_training.py --observation-type topopt_game --algorithm-name PPO --training-phase naive