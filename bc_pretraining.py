import pickle
import numpy as np
from sogym.env import sogym
from sogym.pretraining import pretrain_agent, ExpertDataSet
import yaml
from datetime import datetime
from sogym.utils import profile_and_analyze,ImageDictExtractor, CustomBoxDense
import torch
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

# Initialize parser
parser = argparse.ArgumentParser(description='Pretrain RL agent with expert dataset.')
parser.add_argument('--dataset-file', type=str, default="expert_dataset_topopt_noperm.pkl",
                    help='Path to the expert dataset file.')
parser.add_argument('--observation-type', type=str, default="topopt_game",
                    help='Type of observation used in the environment.')
parser.add_argument('--algorithm-name', type=str, default="SAC",
                    help='Algorithm to use for training (SAC, PPO, TD3).')
parser.add_argument('--algorithm-config-file', type=str, default="algorithms.yaml",
                    help='Path to the algorithm configuration file.')

parser.add_argument('--resume',type = bool, default=False)

parser.add_argument('--resumepath', type=str, default="")

parser.add_argument('--epochs', type=int, default=100)

# Add more arguments as needed

args = parser.parse_args()

def main():
    # Load the expert dataset
    dataset_file = args.dataset_file
    observation_type = args.observation_type
    algorithm_name = args.algorithm_name  # or "TD3"
    algorithm_config_file = args.algorithm_config_file

    chosen_policy = chosen_policy = "MlpPolicy" if observation_type == 'box_dense' else "MultiInputPolicy"
    feature_extractor = ImageDictExtractor if observation_type == 'image' or observation_type == 'topopt_game' else CustomBoxDense

    with open(dataset_file, 'rb') as f:
        data = pickle.load(f)
    expert_observations = data['expert_observations']
    expert_actions = data['expert_actions']

    # Set up the environment
    train_env = sogym(mode='train', observation_type=observation_type, vol_constraint_type='hard', resolution=50, check_connectivity=True)
    eval_env = sogym(mode='test', observation_type=observation_type, vol_constraint_type='hard', resolution=50, check_connectivity=False)

    # Set up the model
    with open(algorithm_config_file, "r") as file:
        config = yaml.safe_load(file)

    algorithm_name = "SAC"
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
        

    # Get the current date and time
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the tb_log_name string
    tb_log_name = f"{algorithm_name}_{current_datetime}"
    experiment_name = f"{algorithm_name}_{observation_type}_{current_datetime}"

    if args.resume:
        print(args.resumepath)
        model.set_parameters(args.resumepath)

    # Pretrain the model
    pretrain_agent(
        model,
        expert_observations,
        expert_actions,
        train_env,
        test_env=eval_env,
        batch_size=4096,
        epochs=args.epochs,
        scheduler_gamma=0.98,
        learning_rate=1.0,
        log_interval=5,
        no_cuda=False,
        seed=1,
        verbose=True,
        test_batch_size=512,
        early_stopping_patience=300,
        plot_curves=True,
        tensorboard_log_dir=f"imitation_tb_logs/{experiment_name}",
        save_path=f"checkpoints/{experiment_name}",
        comet_ml_api_key="No20MKxPKu7vWLOUQCFBRO8mo",
        comet_ml_project_name="pretraining_rl",
        comet_ml_experiment_name=experiment_name,
        eval_freq=5,
        l2_reg_strength=0.001,
    )

if __name__ == "__main__":
    main()
