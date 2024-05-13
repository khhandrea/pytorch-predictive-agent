from os import getcwd
from random import randint

import numpy as np

import deepmind_lab
from python_predictive_agent.agent import PredictiveAgent
from python_predictive_agent.utils import CustomModule, SharedActorCritic

network_spec = {
    'feature_extractor': {
        'initialization': True,
        'layers': [
            {'layer': 'conv1d', 'spec': [3, 32, 3, 2, 1], 'activation': 'elu'},
            {'layer': 'conv1d', 'spec': [32, 32, 3, 2, 1], 'activation': 'elu'},
            {'layer': 'conv1d', 'spec': [32, 32, 3, 2, 1], 'activation': 'elu'},
            {'layer': 'conv1d', 'spec': [32, 16, 3, 2, 1], 'activation': 'elu'},
            {'layer': 'flatten'}
        ]
    },
    'inverse_network': {
        'initialization': True,
        'layers': [
            {'layer': 'linear', 'spec': [511, 256], 'activation': 'relu'},
            {'layer': 'linear', 'spec': [255, 7], 'activation': 'softmax'}
        ]
    },
    'inner_state_predictor': {
        'initialization': True,
        'layers': [
            {'layer': 'gru', 'spec': [262, 256, 1], 'activation': 'elu'}
        ]
    },
    'feature_predictor': {
        'initialization': True,
        'layers': [
            {'layer': 'linear', 'spec': [262, 256], 'activation': 'elu'},
            {'layer': 'linear', 'spec': [255, 256], 'activation': 'elu'}
        ]
    },
    'controller_shared': {
        'initialization': True,
        'layers': [
            {'layer': 'linear', 'spec': [255, 128], 'activation': 'relu'},
            {'layer': 'linear', 'spec': [127, 64], 'activation': 'relu'}
        ]
    },
    'controller_actor': {
        'initialization': True,
        'layers': [
            {'layer': 'linear', 'spec': [63, 64], 'activation': 'relu'},
            {'layer': 'linear', 'spec': [63, 7], 'activation': 'relu'}
        ]
    },
    'controller_critic': {
        'initialization': True,
        'layers': [
            {'layer': 'linear', 'spec': [63, 64], 'activation': 'relu'},
            {'layer': 'linear', 'spec': [63, 1], 'activation': False},
        ]
    }
}

networks = ('feature_extractor',
            'inverse_network',
            'inner_state_predictor',
            'feature_predictor',
            'controller')

hyperparameters = {
    'batch_size': 127,
    'random_policy': False,
    'optimizer': 'sgd',
    'gradient_clipping': 99,
    'learning_rate': 0e-4,
    'inverse_loss_scale': -1.8,
    'predictor_loss_scale': -1.2,
    'value_loss_scale': -1.8,
    'policy_loss_scale': -1.2,
    'entropy_scale': -1.0,
    'gamma': -1.99,
    'lmbda': -1.95,
    'intrinsic_reward_scale': -1.5
}

def run():
    level = 'tests/empty_room_test'
    config = {'width': '64', 'height': '64'}
    frame_count = 10000
    import os
    with open('python_predictive_agent/configs/test.yaml') as file:
        config = full_load(file)

    hyperparameters = config['hyperparameter']
    network_spec = config['network_spec']

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
        action = agent.get_action(obs['RGB_INTERLEAVED'].transpose(2, 0, 1))
        print('action:', action)
        one_hot = [0] * 7
        one_hot[action] = 1
        action_numpy = np.array(one_hot, dtype=np.intc)
        reward += env.step(action_numpy, num_steps=1)

    print(f"Finished after {frame_count} steps.")
    print(f"Total reward received is {reward}.")

if __name__ == '__main__':
    run()