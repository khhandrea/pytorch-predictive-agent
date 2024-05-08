from argparse import ArgumentParser
from datetime import datetime
import os
from shutil import copy
from time import sleep
from yaml import full_load

from torch import load, multiprocessing as mp
from torch.multiprocessing import spawn
from torch.utils.tensorboard import SummaryWriter

from train import train
from utils import CustomModule, ProgressFormatter, SharedActorCritic
from utils import append_to_csv, save_module

def get_config_path() -> str:
    argument_parser = ArgumentParser(prog="Predictive navigation agent RL framework",
                                     description="RL agent with predictive module")
    argument_parser.add_argument('--config', 
                                 required=True,
                                 default='configs/test.yaml',
                                 help="A configuration file path")
    config_path = argument_parser.parse_args().config
    return config_path

def get_configuration(config_path: str) -> dict:
    with open(config_path) as f:
        config = full_load(f)
    return config

def main() -> None:
    config_path = get_config_path()
    config = get_configuration(config_path)

    experiment = config['experiment']
    load_path = config['load_path']
    network_spec = config['network_spec']

    formatted_time = datetime.now().strftime("%y%m%dT%H%M%S")
    experiment_name = f"{formatted_time}_{experiment['name']}"
    print('Experiment name:', experiment_name)
    print('Description:', experiment['description'])

    networks = ('feature_extractor',
                'inverse_network',
                'inner_state_predictor',
                'feature_predictor',
                'controller')

    # Make result directories
    if experiment['save_log'] or experiment['save_trajectory'] or experiment['save_checkpoints']:
        experiment_dir = os.path.join('experiment_results', experiment_name)
        os.mkdir(experiment_dir)
        copy(config_path, os.path.join(experiment_dir, 'config.yaml'))
    if experiment['save_log']:
        os.mkdir(os.path.join(experiment_dir, 'log'))
        log_writer = SummaryWriter(os.path.join(experiment_dir, 'log'))
    if experiment['save_checkpoints']:
        os.mkdir(os.path.join(experiment_dir, 'checkpoints'))
        for network in networks:
            os.mkdir(os.path.join(experiment_dir, 'checkpoints', network))
    if experiment['save_trajectory']:
        os.mkdir(os.path.join(experiment_dir, 'coordinates'))

    # Multiprocessing configuration
    mp.set_start_method('spawn')
    cpu_num = experiment['cpu_num']
    if cpu_num == 0:
        cpu_num = mp.cpu_count()
    print('Subrocess num:', cpu_num)
    
    # Initialize global network
    global_networks = {}
    for network in networks[:-1]:
        global_networks[network] = CustomModule(config['network_spec'][network])
    global_networks['controller'] = SharedActorCritic(
        shared_network=CustomModule(network_spec['controller_shared']),
        actor_network=CustomModule(network_spec['controller_actor']),
        critic_network=CustomModule(network_spec['controller_critic'])
    )

    # Load and share global networks
    for network in networks:
        if load_path[network]:
            global_networks[network].load_state_dict(load(load_path[network]))
            print(f'load {network} from {load_path[network]}')
        global_networks[network].share_memory()

    # Mutiprocessing
    queue = mp.Queue()
    config['environment']['step_max'] = config['hyperparameter']['batch_size'] * experiment['iteration_max'] / cpu_num
    print('Iteration:', experiment['iteration_max'])
    trainer_args = (config['environment'],
                    network_spec,
                    config['hyperparameter'],
                    queue,
                    global_networks
                    )
    mp_context = spawn(fn=train, args=trainer_args, nprocs=cpu_num, daemon=True, join=False)
    print('Spawn complete')

    # Receiving data
    progress = ProgressFormatter()
    iteration = 1
    while any(process.is_alive() for process in mp_context.processes) or not queue.empty():
        if not queue.empty():
            data = queue.get()
            iteration += 1

            # Save coordinates
            if experiment['save_trajectory']:
                coord_dir = os.path.join(experiment_dir, 'coordinates')
                filename = 'process_' + str(data['index']) + '.csv'
                append_to_csv(data['coordinates'], coord_dir, filename)
            del data['coordinates']
            del data['index']

            # Save tensorboard
            if experiment['save_log']:
                for value in data:
                    log_writer.add_scalar(value, data[value], iteration)

            # Print progress
            if iteration % experiment['progress_interval'] == 0:
                data = {'iteration': iteration, **data} # Put iteration at first
                progress.print(data)

            # Save parameter checkpoints
            if experiment['save_checkpoints'] and (iteration % experiment['save_interval'] == 0):
                for network in networks:
                    save_dir = os.path.join(experiment_dir, 'checkpoints', network)
                    file_name = os.path.join(f'step-{iteration}.pt')
                    save_module(global_networks[network], save_dir, file_name)
        else:
            sleep(0.1)

    mp_context.join()
    if experiment['save_log']:
        log_writer.close()

if __name__ == '__main__':
    main()