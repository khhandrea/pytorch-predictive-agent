from argparse import ArgumentParser
from random import randint

import numpy as np

import deepmind_lab

class RandomAgent:
    def __init__(self, action_spec):
        self.action_spec = action_spec
        self.action_count = len(action_spec)

    def step(self):
        action_choice = randint(0, self.action_count - 1)
        action_amount = randint(self.action_spec[action_choice]['min'],
                                self.action_spec[action_choice]['max'])
        action = np.zeros(self.action_count, dtype=np.intc)
        action[action_choice] = action_amount
        return action

def run():
    level = 'tests/empty_room_test'
    config = {'width': '80', 'height': '80'}
    frame_count = 1000

    env = deepmind_lab.Lab(level, ['RGB_INTERLEAVED'], config=config)

    env.reset()

    reward = 0 
    agent = RandomAgent(env.action_spec())
    for _ in range(frame_count):
        if not env.is_running():
            print('Environment stopped early')
            env.reset()
            agent.reset()
        action = agent.step()
        reward += env.step(action, num_steps=1)

    print(f"Finished after {frame_count} steps.")
    print(f"Total reward received is {reward}.")

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    run()