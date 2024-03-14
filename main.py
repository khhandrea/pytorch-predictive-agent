from argparse import ArgumentParser
from time import sleep
from datetime import datetime
from yaml import full_load

from torch import load, multiprocessing as mp

from environments import MovingImageEnvironment
from trainer import train
from utils import copy_file, initialize_custom_model
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
    formatted_time = datetime.now().strftime('%y%m%dT%H%M%S')
    experiment_name = f"{formatted_time}_{experiment['name']}.yaml"
    print('Experiment name:', experiment_name)
    print('Description:', experiment['description'])

    if not experiment['skip_log']:
        copy_file(config_path, 'config_logs', experiment_name)

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
        global_networks[network] = initialize_custom_model(config['network_spec'][network])
    global_networks['controller'] = SharedActorCritic(
        shared_network=initialize_custom_model(config['network_spec']['controller_shared']),
        actor_network=initialize_custom_model(config['network_spec']['controller_actor']),
        critic_network=initialize_custom_model(config['network_spec']['controller_critic'])
    )

    # Load and share global networks
    for network in networks:
        if load_path[network]:
            global_networks[network].load_state_dict(load(load_path[network]))
        global_networks[network].share_memory()

    # Multiprocessing
    mp.set_start_method('spawn')
    cpu_num = experiment['cpu_num']
    if cpu_num == 0:
        cpu_num = mp.cpu_count()

    env = MovingImageEnvironment
    env_args = config['environment']
    hyperparameter = config['hyperparameter']
    queue = mp.Queue()
    trainer_args = (env, env_args, hyperparameter, queue, global_networks)

    processes = []
    for _ in range(cpu_num):
        process = mp.Process(target=train, args=trainer_args)
        process.start()
        processes.append(process)

    # Receiving data
    while any(process.is_alive() for process in processes) or not queue.empty():
        if not queue.empty():
            data = queue.get()
            print("Got data from the queue:", data)
            # save
            # log
            # print progress
        else:
            sleep(0.1)

    for process in processes:
        process.join()