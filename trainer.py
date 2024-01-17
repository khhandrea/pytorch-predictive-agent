from typing import Dict, Any

from gymnasium.utils.env_checker import check_env
from gymnasium import Env

from agent import PredictiveAgent
from utils import LogWriter

class Trainer:
    def __init__(self, 
            env: Env, 
            random_policy: bool=False,
            description: str='default'):
        self._env = env
        self._agent = PredictiveAgent(
            action_space=self._env.action_space, 
            random_policy=random_policy)
        self._log_writer = LogWriter(
            env_name=self._env.__class__.__name__,
            description=description)

        check_env(self._env, skip_render_check=True)

    def train(self):
        observation, info = self._env.reset()
        extrinsic_reward = 0
        terminated = truncated = False
        
        step = 0
        print(f"| {'step':<12} | {'inverse':>12} | {'predictor':>12} | {'policy':>12} | {'value':>12} |")
        while not (terminated or truncated):
            action, values = self._agent.get_action(observation, extrinsic_reward, None)
            observation, extrinsic_reward, terminated, truncated, info = self._env.step(action)

            values['reward/extrinsic_reward'] = extrinsic_reward
            self._log_writer.write(values, step)

            if step % 1000 == 0:
                print(f"| {step:<12} "
                      + f"| {values['loss/inverse_loss']:>12.4f} "
                      + f"| {values['loss/predictor_loss']:>12.4f} "
                      + f"| {values['loss/policy_loss']:>12.4f} "
                      + f"| {values['loss/value_loss']:>12.4f} |")
            step += 1
        print(f'{terminated} or {truncated} at step {step}')

        self._log_writer.close()