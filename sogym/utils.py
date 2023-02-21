from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
import matplotlib.pyplot as plt

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