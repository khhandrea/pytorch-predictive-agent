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
    parser.add_argument('--skip_log', action='store_true', help="The log won't be written")
    parser.add_argument('--progress_interval', type=int, default=2000, help="Per how many steps to print the progress")
    parser.add_argument('--agent_speed', type=int, default=5, help="Set the agent's moving speed")
    parser.add_argument('--step_max', type=int, default=1000, help="Max step of one episode")
    parser.add_argument('--description', default='default', help="A name of the experiment")
    parser.add_argument('--save_interval', type=int, default=50000, help="Per how many steps to save the model")
    parser.add_argument('--skip_save', action='store_true', help="The model won’t be saved")
    parser.add_argument('--load', default=None, help="Load the networks’ parameters from the entire path")
    parser.add_argument('--load_inverse', default=None, help="Load feature extractor inverse network parameters from the specific file")
    parser.add_argument('--load_predictor', default=None, help="Load predictor network parameters from the specific file")
    parser.add_argument('--load_controller', default=None, help="Load controller network parameters from the specific file")
    args = parser.parse_args()

    render_mode = 'none'
    if args.render_window:
        render_mode = 'human'

    env = LinearSpectrumEnvironment(
        render_mode=render_mode,
        agent_speed=args.agent_speed,
        step_max=args.step_max)
    trainer = Trainer(
        env,
        random_policy=args.random_policy,
        description=args.description,
        skip_log=args.skip_log,
        progress_interval=args.progress_interval,
        save_interval=args.save_interval,
        skip_save=args.skip_save,
        load_args=(args.load, args.load_inverse, args.load_predictor, args.load_controller)
        )
    trainer.train()