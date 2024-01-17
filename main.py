from argparse import ArgumentParser

from environments import LinearSpectrumEnvironment
from trainer import Trainer

if __name__ == '__main__':
    parser = ArgumentParser(
        prog='Predict RL framework',
        description="RL Agent with predict module"
        )
    parser.add_argument('--random_policy', action='store_true', help="The agent chooses a random action")
    parser.add_argument('--render_window', action='store_true', help="The window is showed")
    parser.add_argument('--agent_speed', type=int, default=5, help="Set the agent's moving speed")
    parser.add_argument('--step_max', type=int, default=1000, help="Max step of one episode")
    parser.add_argument('--description', default='default', help="A name of the experiment")
    args = parser.parse_args()

    render_mode = None
    if args.render_window:
        render_mode = 'human'

    env = LinearSpectrumEnvironment(
        render_mode=render_mode,
        agent_speed=args.agent_speed,
        step_max=args.step_max)
    trainer = Trainer(
        env,
        random_policy=args.random_policy,
        description=args.description
        )
    trainer.train()