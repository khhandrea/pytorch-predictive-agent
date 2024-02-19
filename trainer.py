from datetime import timedelta, datetime
from time import time

from gymnasium.utils.env_checker import check_env
from gymnasium import Env

from agent import PredictiveAgent
from utils import copy_file, LogWriter

class Trainer:
    def __init__(self, 
            env: Env, 
            config_path: str,
            random_policy: bool,
            description: str,
            skip_log: bool,
            progress_interval: int,
            save_interval: int,
            skip_save: bool,
            load_args: tuple[str, str, str, str],
            device: str,
            feature_size: int,
            hidden_state_size: int,
            module_args: tuple[str, str, str, str, str],
            lr_args: tuple[float, float, float],
            gamma: float,
            policy_discount: float
            ):
        self._env = env
        check_env(self._env, skip_render_check=True)

        formatted_time = datetime.now().strftime('%y%m%dT%H%M%S')
        path = f'{self._env.__class__.__name__}/{formatted_time}_{description}'
        print(f"path: {path}")

        if not skip_log:
            copy_file(config_path, 'config_logs', f'{formatted_time}_{description}.yaml')

        self._agent = PredictiveAgent(
            observation_space=self._env.observation_space,
            action_space=self._env.action_space, 
            random_policy=random_policy,
            path=path,
            device=device,
            feature_size=feature_size,
            hidden_state_size=hidden_state_size,
            module_args=module_args,
            lr_args=lr_args,
            policy_discount=policy_discount,
            gamma=gamma,
        )
        self._save_interval = save_interval
        self._skip_save = skip_save
        self._agent.load(load_args)
        self._log_writer = LogWriter(
            path=path,
            skip_log=skip_log)
        self._progress_interval = progress_interval
        self._checkpoint_time = 0


    def _print_progress(self,
                        first, 
                        step=None, 
                        inverse=None, 
                        predictor=None, 
                        policy=None, 
                        value=None):
        if first:
            self._checkpoint_time = time()
            print(f"| {'step':>12} | {'(acc)inverse':>12} | {'(avg)pred':>12} | {'(avg)policy':>12} | {'(avg)value':>12} | {'interval':>12} |")
        else:
            interval = str(timedelta(seconds = time() - self._checkpoint_time)).split('.')[0]
            print(f"| {step:>12} | {inverse:>12.4f} | {predictor:>12.4f} | {policy:>12.4f} | {value:>12.4f} | {interval:>12} |")
            self._checkpoint_time = time()
            
    def train(self):
        observation, info = self._env.reset()
        terminated = truncated = False
        
        self._print_progress(first=True)
        step = 1
        extrinsic_reward = 0
        inverse_accuracy = sum_pred = sum_policy = sum_value = 0.
        values = {}
        start_time = time()
        while not (terminated or truncated):
            action, values, inverse_correct = self._agent.get_action(observation, extrinsic_reward)
            observation, extrinsic_reward, terminated, truncated, info = self._env.step(action)

            values['reward/extrinsic_reward'] = extrinsic_reward
            values['exploration/state_x'] = self._env.coordinate[0]
            values['exploration/state_y'] = self._env.coordinate[1]
            self._log_writer.write(values, step)
            inverse_accuracy += 1 if inverse_correct else 0
            sum_pred += values['icm/predictor_loss']
            sum_policy += values['controller/policy_loss']
            sum_value += values['controller/value_loss']

            # Print the progress periodically
            if step % self._progress_interval == 0:
                self._print_progress(
                    False,
                    step,
                    inverse_accuracy / self._progress_interval,
                    sum_pred / self._progress_interval,
                    sum_policy / self._progress_interval,
                    sum_value / self._progress_interval)
                inverse_accuracy = sum_pred = sum_policy = sum_value = 0.

            # Save the model periodically
            if step % self._save_interval == 0:
                if not self._skip_save:
                    self._agent.save(f'step-{step}')
                    
            step += 1
        formatted_time = str(timedelta(seconds = time() - start_time)).split('.')[0]
        print(f"{'terminated' if terminated else 'truncated'} at step {step - 1} after {formatted_time}")

        self._log_writer.close()