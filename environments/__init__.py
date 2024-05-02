from gymnasium.envs.registration import register

from environments.linear_spectrum_environment import LinearSpectrumEnvironment
# from environments.circular_spectrum_environment import CircularSpectrumEnvironment
from environments.moving_image_environment import MovingImageEnvironment

register(
    id="pytorch-predictive-agent/LinearSpectrumEnvironment-v0",
    entry_point="environments.linear_spectrum_environment:LinearSpectrumEnvironment",
)

register(
    id="pytorch-predictive-agent/MovingImageEnvironment-v0",
    entry_point="environments.moving_image_environment:MovingImageEnvironment",
)