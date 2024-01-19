from datetime import timedelta
from time import time

from gymnasium.utils.env_checker import check_env
from gymnasium import Env

from agent import PredictiveAgent
from utils import LogWriter

class Trainer:
    def __init__(self, 
            env: Env, 
            random_policy: bool=False,
            description: str='default',
            skip_log: bool=False,
            progress_interval: int=2000):
        self._env = env
        self._agent = PredictiveAgent(
            observation_space=self._env.observation_space,
            action_space=self._env.action_space, 
            random_policy=random_policy)
        self._log_writer = LogWriter(
            env_name=self._env.__class__.__name__,
            description=description,
            skip_log=skip_log)
        self._progress_interval = progress_interval

        check_env(self._env, skip_render_check=True)

    def _print_progress(self,
                        first, 
                        step=None, 
                        inverse=None, 
                        predictor=None, 
                        policy=None, 
                        value=None,
                        interval=None):
        if first:
            print(f"| {'step':>12} | {'inverse':>10} | {'predictor':>10} | {'policy':>10} | {'value':>10} | {'interval':>12} |")
        else:
            print(f"| {step:>12} | {inverse:>10.4f} | {predictor:>10.4f} | {policy:>10.4f} | {value:>10.4f} | {interval:>12} |")
            
    def train(self):
        observation, info = self._env.reset()
        extrinsic_reward = 0
        terminated = truncated = False
        
        self._print_progress(first=True)
        step = 0
        values = {}
        start_time = time()
        checkpoint_time = start_time
        while not (terminated or truncated):
            action, values = self._agent.get_action(observation, extrinsic_reward, None)
            observation, extrinsic_reward, terminated, truncated, info = self._env.step(action)

            values['reward/extrinsic_reward'] = extrinsic_reward
            self._log_writer.write(values, step)

            if step % self._progress_interval == 0:
                interval = str(timedelta(seconds = time() - checkpoint_time)).split('.')[0]
                self._print_progress(
                    False,
                    step,
                    values['loss/inverse_loss'],
                    values['loss/predictor_loss'],
                    values['loss/policy_loss'],
                    values['loss/value_loss'],
                    interval)
                checkpoint_time = time()
            step += 1
        interval = str(timedelta(seconds = time() - checkpoint_time)).split('.')[0]
        self._print_progress(
            False,
            step - 1,
            values['loss/inverse_loss'],
            values['loss/predictor_loss'],
            values['loss/policy_loss'],
            values['loss/value_loss'],
            interval)
        print(f"{'terminated' if terminated else 'truncated'} at step {step - 1}")
        formatted_time = timedelta(seconds = time() - start_time)
        print(f"Total time: {formatted_time}")

        self._log_writer.close()