from gymnasium.utils.env_checker import check_env

class Trainer:
    def __init__(self, env):
        print(check_env(env))
        self._env = env

    def train(self):
        print(self._env.action_space)
        print(self._env.observation_space)