from argparse import ArgumentParser
from yaml import full_load

from environments import MovingImageEnvironment
from trainer import Trainer

if __name__ == '__main__':
    # Configuration
    argument_parser = ArgumentParser(
        prog="Predictive navigation agent RL framework",
        description="RL agent with predictive module"
    )
    argument_parser.add_argument('--config', default='config.conf', help="A configuration file to set configurations")
    args = argument_parser.parse_args()
    with open(args.config) as f:
        config = full_load(f)

    # Preprocessing config
    render_mode = 'human' if config['visualize'] else 'none'
    feature_extractor_inverse_lr = float(config['feature_extractor_inverse_lr'])
    predictor_lr = float(config['predictor_lr'])
    controller_lr = float(config['controller_lr'])
    load_args = (
        config['load'], 
        config['load_inverse'], 
        config['load_predictor'], 
        config['load_controller'])
    lr_args = (
        feature_extractor_inverse_lr, 
        predictor_lr, 
        controller_lr)
    
    # Experiment
    env = MovingImageEnvironment(
        render_mode=render_mode,
        agent_speed=config['agent_speed'],
        step_max=config['step_max'],
        noise_scale=config['noise_scale'],
        )
    trainer = Trainer(
        env,
        random_policy=config['random_policy'],
        description=config['description'],
        skip_log=config['skip_log'],
        progress_interval=config['progress_interval'],
        save_interval=config['save_interval'],
        skip_save=config['skip_save'],
        load_args=load_args,
        device=config['device'],
        lr_args=lr_args,
        policy_discount=config['policy_discount'],
        gamma=config['gamma'],
        hidden_state_size=config['hidden_state_size'],
        feature_size=config['feature_size'],
        )
    trainer.train()