import gymnasium as gym
import environments

if __name__ == '__main__':
    env = gym.make(id='pytorch-predictive-agent/MovingImageEnvironment-v0',
                   render_mode='none',
                   agent_speed=10,
                   step_max=10000,
                   noise_scale=3.0
                   )
