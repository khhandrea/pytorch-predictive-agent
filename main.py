from argparse import ArgumentParser
from configparser import ConfigParser

from environments import LinearSpectrumEnvironment
from trainer import Trainer

if __name__ == '__main__':
    # Configuration
    config = ConfigParser()
    config.read('config.conf')
    config.sections()
    config = config['DEFAULT']

    description = config.get('description')
    load = config.get('load')
    load_inverse = config.get('load_inverse')
    load_predictor = config.get('load_predictor')
    load_controller = config.get('load_controller')
    device = config.get('device')
    visualize = config.getboolean('visualize')
    skip_log = config.getboolean('skip_log')
    skip_save = config.getboolean('skip_save')
    save_interval = config.getint('save_interval')

    step_max = config.getint('step_max')
    progress_interval = config.getint('progress_interval')
    agent_speed = config.getint('agent_speed')
    random_policy = config.getboolean('random_policy')

    feature_extractor_inverse_lr = config.getfloat('feature_extractor_inverse_lr') 
    predictor_lr = config.getfloat('predictor_lr') 
    controller_lr = config.getfloat('controller_lr') 
    hidden_state_size = config.getint('hidden_state_size') 
    feature_size = config.getint('feature_size') 
    predictor_RNN_num_layers = config.getint('predictor_RNN_num_layers') 
    feature_extractor_layerwise_shape = config.get('feature_extractor_layerwise_shape') 
    inverse_network_layerwise_shape = config.get('inverse_network_layerwise_shape') 
    
    render_mode = 'human' if visualize else 'none'
    feature_extractor_layerwise_shape = tuple(map(int, feature_extractor_layerwise_shape.split(',')))
    inverse_network_layerwise_shape = tuple(map(int, inverse_network_layerwise_shape.split(',')))
    print(feature_extractor_layerwise_shape)
    print(inverse_network_layerwise_shape)

    # Experiment

    env = LinearSpectrumEnvironment(
        render_mode=render_mode,
        agent_speed=agent_speed,
        step_max=step_max)
    trainer = Trainer(
        env,
        random_policy=random_policy,
        description=description,
        skip_log=skip_log,
        progress_interval=progress_interval,
        save_interval=save_interval,
        skip_save=skip_save,
        load_args=(load, load_inverse, load_predictor, load_controller),
        device=device,
        lr_args=(feature_extractor_inverse_lr, predictor_lr, controller_lr),
        hidden_state_size=hidden_state_size,
        feature_size=feature_size,
        predictor_RNN_num_layers=predictor_RNN_num_layers,
        feature_extractor_layerwise_shape=feature_extractor_layerwise_shape,
        inverse_network_layerwise_shape=inverse_network_layerwise_shape
        )
    trainer.train()