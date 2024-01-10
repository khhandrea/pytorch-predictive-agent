from gymnasium.envs.registration import register

from environments.linear_spectrum_environment import LinearSpectrumEnvironment
# from environments.circular_spectrum_environment import CircularSpectrumEnvironment

register(
    id="environments/LinearSpectrumEnvironment",
    entry_point="python-predict-agent.environments:LinearSpectrumEnvironment",
    max_episode_steps=300,
)