from argparse import ArgumentParser
from datetime import datetime
import os
from shutil import copy
from time import sleep
from traceback import print_exc
from yaml import full_load

import torch
from torch import nn
from torch import set_default_device, multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from environments import MovingImageEnvironment
from train import train
from utils import CustomModule, SharedActorCritic

def get_config_path() -> str:
    argument_parser = ArgumentParser(
        prog="Predictive navigation agent RL framework",
        description="RL agent with predictive module"
    )
    argument_parser.add_argument('--config', required=True, default='configs/test.yaml', help="A configuration file path")
    config_path = argument_parser.parse_args().config
    return config_path

def get_configuration(config_path: str) -> dict:
    with open(config_path) as f:
        config = full_load(f)
    return config

def save_module(
    module: nn.Module,
    directory: str,
    file_name: str
) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(
        module.state_dict(),
        os.path.join(directory, file_name)
    )

def main() -> None:
    config_path = get_config_path()
    config = get_configuration(config_path)

    experiment = config['experiment']
    load_path = config['load_path']
    training_spec = config['training_spec']
    network_spec = config['network_spec']

    formatted_time = datetime.now().strftime("%y%m%dT%H%M%S")
    experiment_name = f"{formatted_time}_{experiment['name']}"
    print('Experiment name:', experiment_name)
    print('Description:', experiment['description'])

    # Save current configuration file to config_logs
    if experiment['save_log']:
        destination_path = os.path.join('config_logs', experiment_name + '.yaml')
        copy(config_path, destination_path)
        log_writer = SummaryWriter(f"logs/{experiment_name}")

    # Multiprocessing configuration
    mp.set_start_method('spawn')
    cpu_num = experiment['cpu_num']
    if cpu_num == 0:
        cpu_num = mp.cpu_count()
    print('Subrocess num:', cpu_num)
    
    # Device configuration
    if training_spec['device'] != 'cpu':
        assert torch.cuda.is_available()
    device = torch.device(training_spec['device'])
    set_default_device(device)
    print("Device:", device)

    # Initialize global network
    networks = (
        'feature_extractor',
        'inverse_network',
        'inner_state_predictor',
        'feature_predictor',
        'controller'
    )
    global_networks = {}
    for network in networks[:-1]:
        global_networks[network] = CustomModule(config['network_spec'][network])
        global_networks[network] = global_networks[network]
    global_networks['controller'] = SharedActorCritic(
        shared_network=CustomModule(network_spec['controller_shared']),
        actor_network=CustomModule(network_spec['controller_actor']),
        critic_network=CustomModule(network_spec['controller_critic'])
    )

    # Load and share global networks
    for network in networks:
        if load_path[network]:
            global_networks[network].load_state_dict(torch.load(load_path[network]))
            print(f'load {network} from {load_path[network]}')
        global_networks[network].share_memory()

    # Multiprocessing
    env_class = MovingImageEnvironment
    env_name = env_class.__name__
    queue = mp.Queue()
    trainer_args = {
        'env_class': env_class,
        'env_args': config['environment'],
        'device': device,
        'network_spec': network_spec,
        'hyperparameter': config['hyperparameter'],
        'queue': queue,
        'global_networks': global_networks,
        'batch_size': training_spec['batch_size']
    }
    processes = []
    for _ in range(cpu_num):
        process = mp.Process(target=train, kwargs=trainer_args)
        process.start()
        processes.append(process)

    # Receiving data
    progress_contents = (
        'icm/inverse_accuracy',
        'icm/predictor_loss',
        'reward/extrinsic_return',
        'controller/entropy',
        'controller/policy_loss',
        'controller/value_loss',
    )
    print('iteration |', ' | '.join(progress_contents))
    iteration = 1
    while any(process.is_alive() for process in processes) or not queue.empty():
        if not queue.empty():
            data = queue.get()
            iteration += 1

            # Log with tensorboard
            if experiment['save_log']:
                for value in data:
                    log_writer.add_scalar(value, data[value], iteration)

            # Print progress
            if iteration % experiment['progress_interval'] == 0:
                print(
                    f'iteration-{iteration:>8} |',
                    ' | '.join(f'{data[content]:>10.2f}' for content in progress_contents)
                )

            # Save checkpoints
            if experiment['save_checkpoints'] and iteration % experiment['save_interval']:
                for network in networks:
                    save_path = os.path.join('checkpoints', env_name, experiment_name, network)
                    file_name = os.path.join(f'step-{iteration}.pt')
                    save_module(global_networks[network], save_path, file_name)
                    
        else:
            try:
                sleep(0.1)
            except (Exception, KeyboardInterrupt):
                for process in processes:
                    process.kill()
                print('Program is killed unintentionally:')
                print_exc()

    for process in processes:
        process.join()
    if experiment['save_log']:
        log_writer.close()

if __name__ == '__main__':
    main()