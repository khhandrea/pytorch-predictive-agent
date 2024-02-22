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
    argument_parser.add_argument('--config', 
                                 default='config.conf', 
                                 help="A configuration file to set configurations")
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
        config['load_feature_extractor'],
        config['load_inverse'], 
        config['load_inner_state_predictor'],
        config['load_feature_predictor'], 
        config['load_controller'])
    module_args = (
        config['feature_extractor_module'],
        config['inverse_module'],
        config['feature_predictor_module'],
        config['inner_state_predictor_module'],
        config['controller_module'],
    )
    lr_args = (
        feature_extractor_inverse_lr, 
        predictor_lr, 
        controller_lr)
    optimizer_args = (
        config['feature_extractor_optimizer'],
        config['predictor_optimizer'],
        config['controller_optimizer']
    )

    # Experiment
    print(config['description'])
    env = MovingImageEnvironment(
        render_mode=render_mode,
        agent_speed=config['agent_speed'],
        step_max=config['step_max'],
        noise_scale=config['noise_scale'],
        )
    trainer = Trainer(
        env,
        config_path=args.config,
        random_policy=config['random_policy'],
        name=config['name'],
        skip_log=config['skip_log'],
        progress_interval=config['progress_interval'],
        save_interval=config['save_interval'],
        inverse_accuracy_batch_size=config['inverse_accuracy_batch_size'],
        skip_save=config['skip_save'],
        load_args=load_args,
        device=config['device'],
        feature_size=config['feature_size'],
        hidden_state_size=config['hidden_state_size'],
        module_args=module_args,
        lr_args=lr_args,
        optimizer_args=optimizer_args,
        policy_discount=config['policy_discount'],
        entropy_discount=config['entropy_discount'],
        intrinsic_reward_discount=config['intrinsic_reward_discount'],
        predictor_loss_discount=config['predictor_loss_discount'],
        gamma=config['gamma'],
        )
    trainer.train()