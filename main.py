from argparse import ArgumentParser

from environments import LinearSpectrumEnvironment
from trainer import Trainer

if __name__ == '__main__':
    parser = ArgumentParser(
        prog='Predict RL framework',
        description="RL Agent with predict module"
        )
    parser.add_argument('--random-policy', action='store_true', help="The agent chooses a random action")
    parser.add_argument('--render_window', action='store_true', help="The window is showed")
    parser.add_argument('--agent_speed', type=int, default=5, help="Set the agent's moving speed")
    args = parser.parse_args()

    if args.render_window:
        render_mode = 'human'
    config = {
        'random_policy': args.random_policy,
    }

    env = LinearSpectrumEnvironment(
        render_mode=render_mode,
        agent_speed=args.agent_speed)
    trainer = Trainer(env, config)
    trainer.train()