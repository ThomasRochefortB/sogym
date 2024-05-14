from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
import matplotlib.pyplot as plt
import torch

# Class defining the callback to log figures in tensorboard:
class FigureRecorderCallback(BaseCallback):
    
    def __init__(self, eval_env, check_freq: int, figure_size=(8, 6), verbose=0):
        super(FigureRecorderCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.eval_env = eval_env
        self.figure_size = figure_size
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Plot values (here a random variable)
            figures = self.training_env.unwrapped.env_method('plot')
            self._log_figures(figures, "training/figure")

            eval_figures = self.eval_env.unwrapped.env_method('plot')
            self._log_figures(eval_figures, "eval/figure")
        
        return True
    
    def _log_figures(self, figures, tag_prefix):
        # Handle the case where figures is a list
        if isinstance(figures, list):
            for i, figure in enumerate(figures):
                if figure:
                    self._log_figure(figure, f"{tag_prefix}_{i}")
        else:
            if figures:
                self._log_figure(figures, tag_prefix)

    def _log_figure(self, figure, tag):
        # Resize figure to ensure consistency
        figure.set_size_inches(self.figure_size)
        self.logger.record(tag, Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
        plt.close(figure)

class MaxRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MaxRewardCallback, self).__init__(verbose)
        self.max_reward = -float('inf')

    def _on_step(self) -> bool:
        # Ensure you access the reward correctly from the environment
        reward = self.training_env.unwrapped.get_attr('reward')[0]
        self.max_reward = max(self.max_reward, reward)
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record('max_reward', self.max_reward)
        self.max_reward = -float('inf')



class GradientNormCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(GradientNormCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # You can add any necessary logic here
        return True

    def _on_rollout_end(self) -> None:
        # Calculate the gradient norm
        total_norm = 0
        for param in self.model.policy.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.logger.record('gradient_norm', total_norm)


class GradientClippingCallback(BaseCallback):
    def __init__(self, clip_value: float, verbose: int = 0):
        super().__init__(verbose)
        self.clip_value = clip_value

    def _on_step(self) -> bool:
        if hasattr(self.model, "policy"):
            torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), self.clip_value)
        return True