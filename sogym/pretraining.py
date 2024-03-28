from torch.utils.data import Dataset, random_split
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import gymnasium as gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class ExpertDataSet(Dataset):
    def __init__(self, expert_observations, expert_actions, env):
        if isinstance(env.observation_space, gym.spaces.Dict):
            # Handle Dict observation space
            self.observations = {}
            for key in env.observation_space.spaces.keys():
                self.observations[key] = th.from_numpy(np.stack([obs[key] for obs in expert_observations]))
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
    expert_observations,
    expert_actions,
    env,
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
    verbose=True,  # Added verbose option
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
                # Handle Dict observation space
                data = {k: v.to(device) for k, v in data.items()}
            else:
                # Handle Box observation space
                data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()

            if isinstance(env.action_space, gym.spaces.Box):
                # A2C/PPO policy outputs actions, values, log_prob
                # SAC/TD3 policy outputs actions only
                if isinstance(student, (A2C, PPO)):
                    action, _, _ = model(data)
                else:
                    # SAC/TD3:
                    action = model(data)
                action_prediction = action.double()
            else:
                # Retrieve the logits for A2C/PPO when using discrete actions
                dist = model.get_distribution(data)
                action_prediction = dist.distribution.logits
                target = target.long()

            loss = criterion(action_prediction, target)
            train_loss += loss.item()
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Calculate gradient norm after clipping
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
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                        grad_norm,
                        current_lr,
                    )
                )

        train_loss /= num_batches
        if verbose:
            print(f"Train set: Average loss: {train_loss:.4f}")

        # Log training loss and gradient norm to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Grad Norm", grad_norm, epoch)

        return train_loss

    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        num_batches = len(test_loader)
        with th.no_grad():
            for data, target in test_loader:
                if isinstance(data, dict):
                    # Handle Dict observation space
                    data = {k: v.to(device) for k, v in data.items()}
                else:
                    # Handle Box observation space
                    data = data.to(device)
                target = target.to(device)

                if isinstance(env.action_space, gym.spaces.Box):
                    # A2C/PPO policy outputs actions, values, log_prob
                    # SAC/TD3 policy outputs actions only
                    if isinstance(student, (A2C, PPO)):
                        action, _, _ = model(data)
                    else:
                        # SAC/TD3:
                        action = model(data)
                    action_prediction = action.double()
                else:
                    # Retrieve the logits for A2C/PPO when using discrete actions
                    dist = model.get_distribution(data)
                    action_prediction = dist.distribution.logits
                    target = target.long()

                test_loss += criterion(action_prediction, target).item()
        test_loss /= num_batches
        if verbose:
            print(f"Test set: Average loss: {test_loss:.4f}")

        # Log test loss to TensorBoard
        writer.add_scalar("Loss/test", test_loss, epoch)

        return test_loss

    expert_dataset = ExpertDataSet(expert_observations, expert_actions, env)

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

    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    no_improvement_count = 0

    if plot_curves:
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Test Loss Curves')
        train_line, = ax.plot([], [], label='Train Loss')
        test_line, = ax.plot([], [], label='Test Loss')
        ax.legend()

    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch, max_grad_norm=10.0)
        test_loss = test(model, device, test_loader)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if plot_curves:
            train_line.set_data(range(1, epoch + 1), train_losses)
            test_line.set_data(range(1, epoch + 1), test_losses)
            ax.relim()
            # add grid lines:
            ax.grid(True)
            ax.autoscale_view(True, True, True)
            fig.canvas.draw()
            fig.canvas.flush_events()

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if no_improvement_count >= early_stopping_patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}")
            break

        scheduler.step()

    student.policy = model

    # Close the SummaryWriter
    writer.close()
