from torch.utils.data import Dataset, random_split
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from comet_ml import Experiment
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
import torch
torch.autograd.set_detect_anomaly(True)
import datetime
import os
import pickle
import numpy as np
from torch.utils.data import Dataset

def load_expert_dataset(chunk_dir, env):
    chunk_files = [os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.endswith('.pkl')]
    
    expert_observations_list = []
    expert_actions_list = []
    
    for chunk_file in chunk_files:
        with open(chunk_file, 'rb') as f:
            chunk_data = pickle.load(f)
            expert_observations_list.append(chunk_data['expert_observations'])
            expert_actions_list.append(chunk_data['expert_actions'])
    
    if isinstance(env.observation_space, gym.spaces.Dict):
        expert_observations = {}
        for key in env.observation_space.spaces.keys():
            expert_observations[key] = np.concatenate([obs[key] for obs in expert_observations_list])
    else:
        expert_observations = np.concatenate(expert_observations_list)
    
    expert_actions = np.concatenate(expert_actions_list)
    
    return ExpertDataSet(expert_observations, expert_actions, env)

class ExpertDataSet(Dataset):
    def __init__(self, expert_observations, expert_actions, env):
        if isinstance(env.observation_space, gym.spaces.Dict):
            # Handle Dict observation space
            self.observations = {}
            for key in env.observation_space.spaces.keys():
                self.observations[key] = th.from_numpy(expert_observations[key])
        else:
            # Handle Box observation space
            self.observations = th.from_numpy(expert_observations)
        
        if isinstance(env.action_space, gym.spaces.Box):
            self.actions = th.from_numpy(expert_actions).double()
        else:
            self.actions = th.from_numpy(expert_actions).long()

    def __getitem__(self, index):
        if isinstance(self.observations, dict):
            # Handle Dict observation space
            obs_dict = {k: v[index] for k, v in self.observations.items()}
            return obs_dict, self.actions[index]
        else:
            # Handle Box observation space
            return self.observations[index], self.actions[index]

    def __len__(self):
        return len(self.actions)


def pretrain_agent(
    student,
    expert_dataset,
    env,
    test_env,
    batch_size=64,
    epochs=1000,
    scheduler_gamma=0.7,
    learning_rate=1.0,
    log_interval=100,
    no_cuda=True,
    seed=1,
    test_batch_size=64,
    early_stopping_patience=10,
    plot_curves=True,
    tensorboard_log_dir="pretraining_bc",
    verbose=True,
    checkpoint_dir=None,
    load_checkpoint=False,
    comet_ml_api_key=None,
    comet_ml_project_name=None,
    comet_ml_experiment_name=None,
    n_eval_episodes=10,
    eval_freq=10,
    l2_reg_strength=0.0,
    max_grad_norm=None,
):
    use_cuda = not no_cuda and th.cuda.is_available()
    th.manual_seed(seed)
    device = th.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    if isinstance(env.action_space, gym.spaces.Box):
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Extract initial policy
    model = student.policy.to(device)

    # Create a SummaryWriter for TensorBoard logging
    writer = SummaryWriter(log_dir=tensorboard_log_dir)

    def train(model, device, train_loader, optimizer, epoch, max_grad_norm):
        model.train()
        train_loss = 0
        num_batches = len(train_loader)

        for batch_idx, (data, target) in enumerate(train_loader):
            if isinstance(data, dict):
                data = {k: v.to(device) for k, v in data.items()}
            else:
                data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            if isinstance(env.action_space, gym.spaces.Box):
                if isinstance(student, (A2C, PPO)):
                    action, _, _ = model(data)
                else:
                    action = model(data)
                action_prediction = action.double()
            else:
                dist = model.get_distribution(data)
                action_prediction = dist.distribution.logits
                target = target.long()

            loss = criterion(action_prediction, target)
            l2_reg_loss = 0.0
            for param in model.parameters():
                l2_reg_loss += torch.norm(param, 2)
            loss += l2_reg_strength * l2_reg_loss
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            grad_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm ** 0.5

            optimizer.step()
            if verbose and batch_idx % log_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tGrad Norm: {:.6f}\tLR: {:.6f}".format(
                        epoch,
                        (batch_idx + 1) * train_loader.batch_size,
                        len(train_loader.dataset),
                        100.0 * (batch_idx + 1) / len(train_loader),
                        loss.item(),
                        grad_norm,
                        current_lr,
                    )
                )

        train_loss /= num_batches
        if verbose:
            print(f"Train set: Average loss: {train_loss:.4f}")
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Grad Norm", grad_norm, epoch)
        if experiment is not None:
            experiment.log_metric("train_loss", train_loss, step=epoch)
            experiment.log_metric("grad_norm", grad_norm, step=epoch)

        return train_loss

    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        test_mae = 0
        num_batches = len(test_loader)
        with th.no_grad():
            for data, target in test_loader:
                if isinstance(data, dict):
                    data = {k: v.to(device) for k, v in data.items()}
                else:
                    data = data.to(device)
                target = target.to(device)

                if isinstance(env.action_space, gym.spaces.Box):
                    if isinstance(student, (A2C, PPO)):
                        action, _, _ = model(data)
                    else:
                        action = model(data)
                    action_prediction = action.double()
                else:
                    dist = model.get_distribution(data)
                    action_prediction = dist.distribution.logits
                    target = target.long()

                test_loss += criterion(action_prediction, target).item()
                if isinstance(env.action_space, gym.spaces.Box):
                    mae = th.mean(th.abs(action_prediction - target))
                else:
                    mae = th.mean(th.abs(th.argmax(action_prediction, dim=1) - target))
                test_mae += mae.item()

        test_loss /= num_batches
        test_mae /= num_batches
        if verbose:
            print(f"Test set: Average loss: {test_loss:.4f}, Average MAE: {test_mae:.4f}")
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("MAE/test", test_mae, epoch)
        if experiment is not None:
            experiment.log_metric("test_loss", test_loss, step=epoch)
            experiment.log_metric("test_mae", test_mae, step=epoch)
            experiment.log_metric("mae", test_mae, step=epoch)

        return test_loss, test_mae

    train_size = int(0.8 * len(expert_dataset))
    test_size = len(expert_dataset) - train_size
    train_expert_dataset, test_expert_dataset = random_split(
        expert_dataset, [train_size, test_size]
    )

    train_loader = th.utils.data.DataLoader(
        dataset=train_expert_dataset, batch_size=batch_size, shuffle=True, **kwargs
    )
    test_loader = th.utils.data.DataLoader(
        dataset=test_expert_dataset,
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs,
    )

    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=l2_reg_strength)
    scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

    if load_checkpoint is not None and checkpoint_dir is not None:
        if load_checkpoint == 'best':
            model_file = 'model_best.pt'
            optimizer_file = 'optimizer_best.pt'
        elif load_checkpoint == 'latest':
            model_file = 'model_latest.pt'
            optimizer_file = 'optimizer_latest.pt'
        else:
            raise ValueError("Invalid load_checkpoint value. Must be 'best' or 'latest'.")

        model_path = os.path.join(checkpoint_dir, model_file)
        optimizer_path = os.path.join(checkpoint_dir, optimizer_file)

        if os.path.exists(model_path) and os.path.exists(optimizer_path):
            student.load(model_path)
            optimizer.load_state_dict(torch.load(optimizer_path))
            if verbose:
                print(f"Loaded model from {model_path}")
                print(f"Loaded optimizer from {optimizer_path}")
        else:
            if verbose:
                print("Checkpoint files not found. Starting training from scratch.")

    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    best_model_path = None
    best_optimizer_path = None
    best_mean_reward = -float('inf')
    early_stopping_counter = 0

    if plot_curves:
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Test Loss Curves')
        train_line, = ax.plot([], [], label='Train Loss')
        test_line, = ax.plot([], [], label='Test Loss')
        ax.legend()

    if comet_ml_api_key is not None:
        experiment = Experiment(api_key=comet_ml_api_key, project_name=comet_ml_project_name)
        if comet_ml_experiment_name is not None:
            experiment.set_name(comet_ml_experiment_name)
    else:
        experiment = None

    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch, max_grad_norm=max_grad_norm)
        test_loss, test_mae = test(model, device, test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if plot_curves:
            train_line.set_data(range(1, epoch + 1), train_losses)
            test_line.set_data(range(1, epoch + 1), test_losses)
            ax.relim()
            ax.grid(True)
            ax.autoscale_view(True, True, True)
            fig.canvas.draw()
            fig.canvas.flush_events()

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_path = os.path.join(checkpoint_dir, f'model_best.pt')
            best_optimizer_path = os.path.join(checkpoint_dir, f'optimizer_best.pt')
            student.save(best_model_path)
            torch.save(optimizer.state_dict(), best_optimizer_path)
            if verbose:
                print(f"Saved best model to {best_model_path}")
                print(f"Saved best optimizer to {best_optimizer_path}")

        latest_model_path = os.path.join(checkpoint_dir, f'model_latest.pt')
        latest_optimizer_path = os.path.join(checkpoint_dir, f'optimizer_latest.pt')
        student.save(latest_model_path)
        torch.save(optimizer.state_dict(), latest_optimizer_path)

        if epoch % eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(student, test_env, n_eval_episodes=n_eval_episodes)
            writer.add_scalar("mean_reward", mean_reward, epoch)
            writer.add_scalar("std_reward", std_reward, epoch)
            if verbose:
                print(f"Epoch {epoch}: Mean reward = {mean_reward:.3f} +/- {std_reward:.3f}")
            if experiment is not None:
                experiment.log_metric("mean_reward", mean_reward, step=epoch)
                experiment.log_metric("std_reward", std_reward, step=epoch)

            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping triggered after {epoch} epochs.")
                    break

        scheduler.step()

    student.policy = model

    writer.close()
    if experiment is not None:
        experiment.end()
