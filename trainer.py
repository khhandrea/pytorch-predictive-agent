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

    def _print_progress(self,
                        first, 
                        step=None, 
                        inverse=None, 
                        predictor=None, 
                        policy=None, 
                        value=None):
        if first:
            print(f"| {'step':<12} | {'inverse':>12} | {'predictor':>12} | {'policy':>12} | {'value':>12} |")
        else:
            print(f"| {step:<12} | {inverse:>12.4f} | {predictor:>12.4f} | {policy:>12.4f} | {value:>12.4f} |")
            
    def train(self):
        observation, info = self._env.reset()
        extrinsic_reward = 0
        terminated = truncated = False
        
        self._print_progress(first=True)
        step = 0
        values = {}
        while not (terminated or truncated):
            action, values = self._agent.get_action(observation, extrinsic_reward, None)
            observation, extrinsic_reward, terminated, truncated, info = self._env.step(action)

            values['reward/extrinsic_reward'] = extrinsic_reward
            self._log_writer.write(values, step)

            if step % 1000 == 0:
                self._print_progress(
                    False,
                    step,
                    values['loss/inverse_loss'],
                    values['loss/predictor_loss'],
                    values['loss/policy_loss'],
                    values['loss/value_loss'])
            step += 1
        self._print_progress(
            False,
            step - 1,
            values['loss/inverse_loss'],
            values['loss/predictor_loss'],
            values['loss/policy_loss'],
            values['loss/value_loss'])
        print(f"{'terminated' if terminated else 'truncated'} at step {step - 1}")
        

        self._log_writer.close()