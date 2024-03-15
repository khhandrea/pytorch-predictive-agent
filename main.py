from argparse import ArgumentParser
from datetime import datetime
import os
from shutil import copy
from time import sleep
from yaml import full_load

import torch
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from environments import MovingImageEnvironment
from trainer import train
from utils import initialize_custom_model
from utils import SharedActorCritic

if __name__ == '__main__':
    # Configuration
    argument_parser = ArgumentParser(
        prog="Predictive navigation agent RL framework",
        description="RL agent with predictive module"
    )
    argument_parser.add_argument('--config', default='configs/test.yaml', help="A configuration file")
    config_path = argument_parser.parse_args().config
    with open(config_path) as f:
        config = full_load(f)

    experiment = config['experiment']
    load_path = config['load_path']
    training_spec = config['training_spec']
    formatted_time = datetime.now().strftime('%y%m%dT%H%M%S')
    experiment_name = f"{formatted_time}_{experiment['name']}"
    if training_spec['device'] != 'cpu':
        assert torch.cuda.is_available()
    device = torch.device(training_spec['device'])
    print('Experiment name:', experiment_name)
    print('Description:', experiment['description'])
    print('Device:', device)

    # Save current configuration file to config_logs
    if experiment['save_log']:
        destination_path = os.path.join('config_logs', experiment_name + '.yaml')
        copy(config_path, destination_path)
        log_writer = SummaryWriter(f"logs/{experiment_name}")

    # Global networks
    networks = (
        'feature_extractor',
        'inverse_network',
        'inner_state_predictor',
        'feature_predictor',
        'controller'
    )
    global_networks = {}
    for network in networks[:-1]:
        global_networks[network] = initialize_custom_model(config['network_spec'][network]).to(device)
    global_networks['controller'] = SharedActorCritic(
        shared_network=initialize_custom_model(config['network_spec']['controller_shared']),
        actor_network=initialize_custom_model(config['network_spec']['controller_actor']),
        critic_network=initialize_custom_model(config['network_spec']['controller_critic'])
    ).to(device)

    # Load and share global networks
    for network in networks:
        if load_path[network]:
            global_networks[network].load_state_dict(torch.load(load_path[network], map_location=device))
            print(f"load {network} from {load_path[network]}")
        global_networks[network].share_memory()

    # Multiprocessing
    # mp.set_start_method('spawn')
    cpu_num = experiment['cpu_num']
    if cpu_num == 0:
        cpu_num = mp.cpu_count()

    env_class = MovingImageEnvironment
    queue = mp.Queue()
    trainer_args = {
        'env_class': env_class,
        'env_args': config['environment'],
        'device': device,
        'hyperparameter': config['hyperparameter'],
        'queue': queue,
        'global_networks': global_networks,
        'batch_size': training_spec['batch_size']
    }

    processes = []
    for _ in range(cpu_num):
        process = mp.Process(target=train, kwargs=trainer_args)
        print(f"debug: let's start {process}")
        process.start()
        print("debug: let's append")
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
    print("iteration |", " | ".join(progress_contents))
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
                    f"iteration-{iteration:>8} |",
                    ' | '.join(f"{data[content]:>10.2f}" for content in progress_contents)
                )

            # Save checkpoints
            if experiment['save_checkpoints'] and iteration % experiment['save_interval']:
                save_path = os.path.join("checkpoints", env_class.__name__, experiment_name)
                for network in networks:
                    save_path = os.path.join(save_path, network)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    torch.save(
                        global_networks[network].state_dict(),
                        os.path.join(save_path, f"step-{iteration}.pt")
                    )
        else:
            sleep(0.1)

    for process in processes:
        process.join()
    if experiment['save_log']:
        log_writer.close()