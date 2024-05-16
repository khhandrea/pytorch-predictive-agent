from gymnasium.envs.registration import register

register(id="pytorch-predictive-agent/LinearSpectrumEnvironment-v0",
         entry_point="environments.linear_spectrum_environment:LinearSpectrumEnvironment")
register(id="pytorch-predictive-agent/MovingImageEnvironment-v0",
         entry_point="environments.moving_image_environment:MovingImageEnvironment")
register(id="pytorch-predictive-agent/VisualCartPole-v0",
         entry_point="environments.visual_cartpole:VisualCartPole")
register(id="pytorch-predictive-agent/DeepmindLab-v0",
         entry_point="environments.deepmind_lab_environment:DeepmindLabEnvironment")