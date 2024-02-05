from gymnasium.envs.registration import register

from environments.linear_spectrum_environment import LinearSpectrumEnvironment
# from environments.circular_spectrum_environment import CircularSpectrumEnvironment
from environments.moving_image_environment import MovingImageEnvironment

register(
    id="environments/LinearSpectrumEnvironment",
    entry_point="python-predictive-agent.environments:LinearSpectrumEnvironment",
    max_episode_steps=300,
)

register(
    id="environments/MovingImageEnvironment",
    entry_point="python-predictive-agent.environments:MovingImageEnvironment",
    max_episode_steps=300
)