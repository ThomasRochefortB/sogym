from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import torch as th
from torch import nn
import matplotlib.pyplot as plt
import cProfile
import pstats
import pandas as pd

def run_episodes(num_episodes, env):
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()  # Replace with your agent's action selection logic
            obs, reward, done, truncated, info = env.step(action)

def profile_and_analyze(num_episodes, env):
    # Create a cProfile profiler
    profiler = cProfile.Profile()

    # Start profiling
    profiler.enable()

    # Run the episodes
    run_episodes(num_episodes, env)

    # Stop profiling
    profiler.disable()

    # Print the profiling statistics
    stats = pstats.Stats(profiler)

    # Extract profiling data
    data = []
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        filename, lineno, func_name = func
        # Concatenate filename, line number, and function name to form the full path
        full_func_path = f"{filename}:{lineno}({func_name})"
        data.append([full_func_path, nc, tt, ct])

    # Create a DataFrame with the full function path
    df = pd.DataFrame(data, columns=['Full Function Path', 'ncalls', 'tottime', 'cumtime'])

    # Include percall calculations
    for i, row in enumerate(data):
        full_func_path, ncalls, tottime, cumtime = row
        percall_tottime = tottime / ncalls
        percall_cumtime = cumtime / ncalls
        # Update row with percall values
        data[i] = [full_func_path, ncalls, tottime, percall_tottime, cumtime, percall_cumtime]

    # Update DataFrame with percall columns
    df = pd.DataFrame(data, columns=['Full Function Path', 'ncalls', 'tottime', 'percall (tottime)', 'cumtime', 'percall (cumtime)'])

    # Sort DataFrame by 'percall (cumtime)' in descending order
    df.sort_values(by='percall (cumtime)', ascending=False, inplace=True)

    # Save DataFrame to CSV file
    df.to_csv('profile.csv', index=False)

    return df

# Class defining the callback to log figures in tensorboard:
class FigureRecorderCallback(BaseCallback):
    
    def __init__(self,check_freq: int, verbose=0):
        super(FigureRecorderCallback, self).__init__(verbose)
        self.check_freq = check_freq
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:

            # Plot values (here a random variable)
            figure = self.training_env.env_method('plot')
            # Close the figure after logging it
            self.logger.record("trajectory/figure", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
            plt.close()
            
        return True
    

class ImageDictExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Dict,drop_rate=0.0,activ_func_string='relu',last_conv_size=64,mlp_size=128):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules

        super(ImageDictExtractor, self).__init__(observation_space, features_dim=1)
        self.drop_rate = drop_rate
        activations = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'leaky_relu': nn.LeakyReLU(),}

        self.activ_func = activations[activ_func_string]
        self.last_conv_size = last_conv_size
        self.mlp_size = mlp_size

        extractors = {}
        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            
            if key =='image':
                #image is channel-first in SB3 convention: (C, H, W)
                #default is 64,128,3
                input_h ,input_w, input_c = subspace.shape[1],subspace.shape[2],subspace.shape[0]
                extractors[key]= nn.Sequential(
                                                nn.Conv2d( input_c, 32, kernel_size=3, stride=2, padding=1), #Out is 64 x 32 x 32
                                                nn.ReLU(),
                                                nn.Dropout(p=self.drop_rate),
                                                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), #Out is 32 x 16 x 64
                                                nn.ReLU(),
                                                nn.Dropout(p=self.drop_rate),
                                                nn.Conv2d(64, self.last_conv_size, kernel_size=3, stride=2, padding=1), #Out is 16 x 8 x 64
                                                nn.ReLU(),
                                                nn.Dropout(p=self.drop_rate),
                                                nn.Flatten(),
                                            )
                total_concat_size+= ((input_w * input_h) // 64)  * self.last_conv_size
            
            elif key == "design_variables" or key=="volume" or key=="n_steps_left" or key =="conditions":
                # run through a simple MLP
                extractors[key] = nn.Sequential(
                                                nn.Linear(subspace.shape[0], self.mlp_size),
                                                self.activ_func,
                                                nn.Dropout(p=self.drop_rate),
                                                nn.Linear(self.mlp_size,self.mlp_size),
                                                self.activ_func,
                                                nn.Dropout(p=self.drop_rate),
                                                nn.Flatten()
                                               )
                total_concat_size += (self.mlp_size)
  
        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)
    

class CustomBoxDense(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, hidden_size: int = 32, noise_scale: float = 0.0,device='cpu',batch_norm=True):
        super().__init__(observation_space, features_dim=hidden_size)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        self.noise_scale = noise_scale
        input_len = observation_space.shape[0]
        self.linear = nn.Sequential(
            nn.Linear(input_len, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Flatten(),
        )
        if batch_norm == False:
            self.linear = nn.Sequential(
                nn.Linear(input_len, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Flatten(),
            )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(observations)




