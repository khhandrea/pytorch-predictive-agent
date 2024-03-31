# Config file description

- **experiment**: experimental informations
    - name (*str*): the name of the result files whould be like '{date}T{time}_{name}'
    - description (*str*): what will be printed to the console at first
    - save_log (*boolean*): whether to save the model evalution metrics
    - save_checkpoints (*boolean*): whether to save the model parameters
    - save_interval (*int*): how many iterations to save once
    - progress_interval (*int*): how many iterations to print to the console once
    - cpu_num (*int*): how many cpus to progress multiprocessing. *Maximal cpu_num if -1*

- **environment**: factors that affect the environments in each cpu
    - render_mode (*str*): *TBD*
    - step_max (*int*): max step for the environment
    - agent_speed (*int*): speed of the agent in the environment
    - noise_scale (*int*): noise scale for the observation

- **load_path**: whether to load the model parameters from the existing file. *not to load if 'False'
    - feature_extractor (*str or 'False'*): load feature_extractor parameters
    - inverse_network (*str of 'False'*): load inverse_network
    - inner_state_predictor (*str of 'False'*): load inner_state_predictor
    - feature_predictor (*str of 'False'*): load feature_predictor
    - controller (*str of 'False'*): load controller

- **hyperparameter**: hyperparameters for model optimization
    - batch_size (*int*): how many steps to update parameters once
    - random_policy (*boolean*): if the agent chooses random actions
    - optimizer (*str*): e.g. 'sgd', 'adam'
    - gradient_clipping (*float*): norm to clip the loss gradient *no gradient clipping if -1*
    - learning_rate (*float*)
    - inverse_loss_scale (*float*): e.g. 0.5 to have the half impact
    - predictor_loss_scale (*float*): e.g. 0.5 to have the half impact
    - value_loss_scale (*float*): e.g. 0.5 to have the half impact
    - policy_loss_scale (*float*): e.g. 0.5 to have the half impact
    - entropy_scale (*float*): e.g. 0.5 to have the half impact
    - gamma (*float*): discount factor for reinforcement learning return
    - lmbda (*float*): factor for reinforcement learning GAE
    - intrinsic_reward_scale (*float*): e.g. 0.5 to have the half impact

- **network_spec**: spec of the neural networks. Each of the spec follows [this](https://github.com/khhandrea/pytorch-custom-module)
    - feature_extractor
    - inverse_network
    - inner_state_predictor
    - controller_shared
    - controller_actor
    - controller_critic