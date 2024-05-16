from pathlib import Path
from yaml import full_load

import gymnasium as gym
import numpy as np

from python_predictive_agent import environments
from python_predictive_agent.agent import PredictiveAgent
from python_predictive_agent.utils import CustomModule, SharedActorCritic, preprocess_observation

networks = ('feature_extractor',
            'inverse_network',
            'inner_state_predictor',
            'feature_predictor',
            'controller')

def run():
    level = 'tests/empty_room_test'
    frame_count = 10000
    with open('python_predictive_agent/configs/test-dml.yaml') as file:
        configs = full_load(file)

    env_config = configs['environment']
    env_config['step_max'] = 1000
    hyperparameters = configs['hyperparameter']
    network_spec = configs['network_spec']

    # Initialize global network
    global_networks = {}
    for network in networks[:-1]:
        global_networks[network] = CustomModule(network_spec[network])
    global_networks['controller'] = SharedActorCritic(
        shared_network=CustomModule(network_spec['controller_shared']),
        actor_network=CustomModule(network_spec['controller_actor']),
        critic_network=CustomModule(network_spec['controller_critic'])
    )

    # Load and share global networks
    for network in networks:
        global_networks[network].share_memory()

    env = gym.make(**env_config)
    agent = PredictiveAgent(action_num=env.action_space.n, # 8
                            network_spec=network_spec,
                            global_networks=global_networks,
                            hyperparameters=hyperparameters)

    observation, _ = env.reset()

    terminated = truncated = False
    extrinsic_return = 0
    total_step = 0
    while not (terminated or truncated):
        observation = preprocess_observation(observation, (64, 64), 255, 0)
        action = agent.get_action(observation)
        print('action:', action)
        next_observation, extrinsic_reward, terminated, truncated, info = env.step(action)
        # replay.add_experience(observation, action, extrinsic_reward, terminated or truncated)
        extrinsic_return += extrinsic_reward
        observation = next_observation

        total_step += 1

    path_result = Path(configs['experiment']['home_directory']) / 'experiment_results' / 'test.txt'
    with path_result.open('a') as f:
        f.write('Hello world!\n')
        f.write('exit\n')
    print(f"writing finish on {path_result}")
    print(f"Finished after {total_step} steps. ({'terminated' if terminated else 'truncated'})")
    print(f"Total reward received is {extrinsic_reward}.")

if __name__ == '__main__':
    run()