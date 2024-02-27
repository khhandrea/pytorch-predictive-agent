from os import getcwd
from random import randint

import numpy as np

import deepmind_lab
from python_predictive_agent.agent import PredictiveAgent

def run():
    level = 'tests/empty_room_test'
    config = {'width': '80', 'height': '80'}
    frame_count = 1000

    env = deepmind_lab.Lab(level, ['RGB_INTERLEAVED'], config=config)

    env.reset()

    reward = 0 
    agent = PredictiveAgent()
    for _ in range(frame_count):
        if not env.is_running():
            print('Environment stopped early')
            env.reset()
            agent.reset()
        obs = env.observations()
        action = agent.get_action(obs['RGB_INTERLEAVED'], reward)
        reward += env.step(action, num_steps=1)

    print(f"Finished after {frame_count} steps.")
    print(f"Total reward received is {reward}.")

if __name__ == '__main__':
    run()