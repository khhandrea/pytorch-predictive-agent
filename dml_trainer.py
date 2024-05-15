from os import getcwd
from random import randint
from yaml import full_load

import numpy as np

import deepmind_lab
from python_predictive_agent.agent import PredictiveAgent
from python_predictive_agent.utils import CustomModule, SharedActorCritic, preprocess_observation

def run():
    level = 'tests/empty_room_test'
    config = {'width': '640', 'height': '480'}
    frame_count = 10000
    with open('python_predictive_agent/configs/test.yaml') as file:
        configs = full_load(file)

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


    env = deepmind_lab.Lab(level, ['RGB_INTERLEAVED'], config=config)
    env.reset()

    reward = 0 
    agent = PredictiveAgent(action_num=len(env.action_spec()), # 7
                            network_spec=network_spec,
                            global_networks=global_networks,
                            hyperparameters=hyperparameters)
    for _ in range(frame_count):
        if not env.is_running():
            print('Environment stopped early')
            env.reset()
            agent.reset()
        obs = env.observations()
        preprocessed_observation = preprocess_observation(obs['RGB_INTERLEAVED'].transpose(2, 0, 1), (64, 64), 255, 0)
        action = agent.get_action(preprocessed_observation)
        print('action:', action)
        one_hot = [0] * 7
        one_hot[action] = 1
        action_numpy = np.array(one_hot, dtype=np.intc)
        reward += env.step(action_numpy, num_steps=1)

    print(f"Finished after {frame_count} steps.")
    print(f"Total reward received is {reward}.")

if __name__ == '__main__':
    run()