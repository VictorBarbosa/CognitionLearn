import yaml
from collections import OrderedDict

# --- YAML Representer for OrderedDict ---
def represent_ordereddict(dumper, data):
    value = []
    for item_key, item_value in data.items():
        node_key = dumper.represent_data(item_key)
        node_value = dumper.represent_data(item_value)
        value.append((node_key, node_value))
    return yaml.nodes.MappingNode(u'tag:yaml.org,2002:map', value)

yaml.add_representer(OrderedDict, represent_ordereddict)

def odict_to_dict(d):
    if isinstance(d, OrderedDict):
        d = dict(d)
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = odict_to_dict(value)
    elif isinstance(d, list):
        d = [odict_to_dict(item) for item in d]
    return d

# --- Input Helper Functions ---
def get_input(prompt, default=None, cast_type=str):
    while True:
        default_text = f" (default: {default})" if default is not None else ""
        try:
            response = input(f"{prompt}{default_text}: ")
            if not response and default is not None:
                return default
            if cast_type == str and response.lower() in ['none', 'null']:
                return None
            return cast_type(response)
        except ValueError:
            print(f"Invalid input. Please enter a value of type {cast_type.__name__}.")

def get_bool(prompt, default=True):
    default_str = 'y' if default else 'n'
    prompt_str = f"{prompt} (y/n, default: {default_str}): "
    while True:
        response = input(prompt_str).lower()
        if not response:
            return default
        if response in ['y', 'yes']:
            return True
        if response in ['n', 'no']:
            return False
        print("Invalid answer. Please enter 'y' for yes or 'n' for no.")

def get_choice(prompt, options, default_index=0):
    print(prompt)
    for i, option in enumerate(options, 1):
        default_marker = " (default)" if i - 1 == default_index else ""
        print(f"  {i}: {option}{default_marker}")
    while True:
        try:
            choice_str = input(f"Choose an option (1-{len(options)}): ")
            if not choice_str:
                return options[default_index]
            choice_index = int(choice_str) - 1
            if 0 <= choice_index < len(options):
                return options[choice_index]
            else:
                print("Invalid option. Try again.")
        except ValueError:
            print("Please enter a number.")

def get_multiple_choices(prompt, options):
    print(prompt)
    print("  0: all")
    for i, option in enumerate(options, 1):
        print(f"  {i}: {option}")
    while True:
        response = input(f"Choose one or more options (comma-separated, e.g., 1,3 or 0 for all): ")
        if not response:
            print("No option selected. Please choose at least one.")
            continue
        if response.strip() == '0':
            return options
        try:
            choices_str = response.split(',')
            selected_indices = [int(s.strip()) - 1 for s in choices_str]
            if all(0 <= i < len(options) for i in selected_indices):
                return list(OrderedDict.fromkeys([options[i] for i in selected_indices]))
            else:
                print("One or more selected numbers are out of range. Try again.")
        except (ValueError, IndexError):
            print("Invalid input. Please enter comma-separated numbers corresponding to the options.")

# --- Hyperparameter & Settings Functions ---

def get_common_network_settings():
    network_settings = OrderedDict()
    network_settings['normalize'] = get_bool("Normalize inputs?", True)
    network_settings['hidden_units'] = get_input("Hidden units", 256, int)
    network_settings['num_layers'] = get_input("Number of layers", 2, int)
    if get_bool("Use Memory (LSTM)?", True):
        network_settings['memory'] = OrderedDict([
            ('sequence_length', get_input("Sequence length", 128, int)),
            ('memory_size', get_input("Memory size", 256, int))
        ])
    return network_settings

def get_common_reward_signals():
    reward_signals = OrderedDict()
    print("\n--- Extrinsic Reward Signal ---")
    extrinsic_settings = OrderedDict([
        ('gamma', get_input("gamma", 0.995, float)),
        ('strength', get_input("strength", 1.0, float)),
        ('network_settings', OrderedDict([
            ('normalize', True),
            ('hidden_units', 128),
            ('num_layers', 2)
        ]))
    ])
    reward_signals['extrinsic'] = extrinsic_settings
    return reward_signals

def get_common_trainer_settings():
    settings = OrderedDict()
    settings['keep_checkpoints'] = get_input("keep_checkpoints", 5, int)
    settings['max_steps'] = get_input("max_steps", 50000000, int)
    settings['time_horizon'] = get_input("time_horizon", 2000, int)
    settings['summary_freq'] = get_input("summary_freq", 5000, int)
    return settings

def get_ppo_params(defaults):
    print("\n--- Configuring PPO ---")
    params = OrderedDict()
    params['batch_size'] = get_input("batch_size", defaults.get('batch_size', 128), int)
    params['buffer_size'] = get_input("buffer_size", defaults.get('buffer_size', 10240), int)
    params['learning_rate'] = get_input("learning_rate", defaults.get('learning_rate', 0.0001), float)
    params['beta'] = get_input("beta", 0.01, float)
    params['epsilon'] = get_input("epsilon", 0.2, float)
    params['lambd'] = get_input("lambd", 0.95, float)
    params['num_epoch'] = get_input("num_epoch", 3, int)
    params['learning_rate_schedule'] = get_choice("learning_rate_schedule", ["linear", "constant"], 0)
    return OrderedDict([('trainer_type', 'ppo'), ('hyperparameters', params)])

def get_sac_params(defaults):
    print("\n--- Configuring SAC ---")
    params = OrderedDict()
    params['learning_rate'] = get_input("learning_rate", defaults.get('learning_rate', 0.0001), float)
    params['learning_rate_schedule'] = get_choice("learning_rate_schedule", ["linear", "constant"], 0)
    params['batch_size'] = get_input("batch_size", defaults.get('batch_size', 128), int)
    params['buffer_size'] = get_input("buffer_size", defaults.get('buffer_size', 10240), int)
    params['buffer_init_steps'] = get_input("buffer_init_steps", 20000, int)
    params['tau'] = get_input("tau", 0.005, float)
    params['steps_per_update'] = get_input("steps_per_update", 1.0, float)
    params['save_replay_buffer'] = get_bool("save_replay_buffer", True)
    params['init_entcoef'] = get_input("init_entcoef", 0.1, float)
    return OrderedDict([('trainer_type', 'sac'), ('hyperparameters', params)])

def get_tdsac_params(defaults):
    print("\n--- Configuring TDSAC ---")
    params = OrderedDict()
    params['learning_rate'] = get_input("learning_rate", defaults.get('learning_rate', 0.0003), float)
    params['learning_rate_schedule'] = get_choice("learning_rate_schedule", ["linear", "constant"], 0)
    params['batch_size'] = get_input("batch_size", defaults.get('batch_size', 64), int)
    params['buffer_size'] = get_input("buffer_size", defaults.get('buffer_size', 50000), int)
    params['buffer_init_steps'] = get_input("buffer_init_steps", 10000, int)
    params['tau'] = get_input("tau", 0.005, float)
    params['steps_per_update'] = get_input("steps_per_update", 1.0, float)
    params['save_replay_buffer'] = get_bool("save_replay_buffer", True)
    params['init_entcoef'] = get_input("init_entcoef", 0.1, float)  # TDSAC tem init_entcoef
    return OrderedDict([('trainer_type', 'tdsac'), ('hyperparameters', params)])

def get_td3_params(defaults):
    print("\n--- Configuring TD3 ---")
    params = OrderedDict()
    params['learning_rate'] = get_input("learning_rate", defaults.get('learning_rate', 0.0003), float)
    params['learning_rate_schedule'] = get_choice("learning_rate_schedule", ["linear", "constant"], 0)
    params['batch_size'] = get_input("batch_size", defaults.get('batch_size', 64), int)
    params['buffer_size'] = get_input("buffer_size", defaults.get('buffer_size', 50000), int)
    params['buffer_init_steps'] = get_input("buffer_init_steps", 10000, int)
    params['tau'] = get_input("tau", 0.005, float)
    params['steps_per_update'] = get_input("steps_per_update", 1.0, float)
    params['save_replay_buffer'] = get_bool("save_replay_buffer", True)
    # TD3 não tem init_entcoef
    params['policy_delay'] = get_input("policy_delay", 2, int)
    return OrderedDict([('trainer_type', 'td3'), ('hyperparameters', params)])

def get_tqc_params(defaults):
    print("\n--- Configuring TQC ---")
    params = OrderedDict()
    params['learning_rate'] = get_input("learning_rate", defaults.get('learning_rate', 0.0003), float)
    params['learning_rate_schedule'] = get_choice("learning_rate_schedule", ["linear", "constant"], 0)
    params['batch_size'] = get_input("batch_size", defaults.get('batch_size', 64), int)
    params['buffer_size'] = get_input("buffer_size", defaults.get('buffer_size', 50000), int)
    params['buffer_init_steps'] = get_input("buffer_init_steps", 10000, int)
    params['tau'] = get_input("tau", 0.005, float)
    params['steps_per_update'] = get_input("steps_per_update", 1.0, float)
    params['save_replay_buffer'] = get_bool("save_replay_buffer", True)
    params['init_entcoef'] = get_input("init_entcoef", 0.1, float)  # TQC tem init_entcoef
    params['n_quantiles'] = get_input("n_quantiles", 25, int)
    params['n_to_drop'] = get_input("n_to_drop", 2, int)
    return OrderedDict([('trainer_type', 'tqc'), ('hyperparameters', params)])

def get_dcac_params(defaults):
    print("\n--- Configuring DCAC ---")
    params = OrderedDict()
    params['learning_rate'] = get_input("learning_rate", defaults.get('learning_rate', 0.0003), float)
    params['learning_rate_schedule'] = get_choice("learning_rate_schedule", ["linear", "constant"], 0)
    params['batch_size'] = get_input("batch_size", defaults.get('batch_size', 64), int)
    params['buffer_size'] = get_input("buffer_size", defaults.get('buffer_size', 50000), int)
    params['buffer_init_steps'] = get_input("buffer_init_steps", 10000, int)
    params['tau'] = get_input("tau", 0.005, float)
    params['steps_per_update'] = get_input("steps_per_update", 1.0, float)
    params['save_replay_buffer'] = get_bool("save_replay_buffer", True)
    params['init_entcoef'] = get_input("init_entcoef", 0.1, float)  # DCAC tem init_entcoef
    params['destructive_threshold'] = get_input("destructive_threshold", 0.0, float)
    return OrderedDict([('trainer_type', 'dcac'), ('hyperparameters', params)])

def get_crossq_params(defaults):
    print("\n--- Configuring CrossQ ---")
    params = OrderedDict()
    params['learning_rate'] = get_input("learning_rate", defaults.get('learning_rate', 0.0003), float)
    params['learning_rate_schedule'] = get_choice("learning_rate_schedule", ["linear", "constant"], 0)
    params['batch_size'] = get_input("batch_size", defaults.get('batch_size', 64), int)
    params['buffer_size'] = get_input("buffer_size", defaults.get('buffer_size', 50000), int)
    params['buffer_init_steps'] = get_input("buffer_init_steps", 10000, int)
    params['tau'] = get_input("tau", 0.005, float)
    params['steps_per_update'] = get_input("steps_per_update", 1.0, float)
    params['save_replay_buffer'] = get_bool("save_replay_buffer", True)
    # CrossQ não tem init_entcoef
    return OrderedDict([('trainer_type', 'crossq'), ('hyperparameters', params)])

def get_drqv2_params(defaults):
    print("\n--- Configuring DrQv2 ---")
    params = OrderedDict()
    params['learning_rate'] = get_input("learning_rate", defaults.get('learning_rate', 0.0003), float)
    params['learning_rate_schedule'] = get_choice("learning_rate_schedule", ["linear", "constant"], 0)
    params['batch_size'] = get_input("batch_size", defaults.get('batch_size', 64), int)
    params['buffer_size'] = get_input("buffer_size", defaults.get('buffer_size', 50000), int)
    params['buffer_init_steps'] = get_input("buffer_init_steps", 10000, int)
    params['tau'] = get_input("tau", 0.005, float)
    params['steps_per_update'] = get_input("steps_per_update", 1.0, float)
    params['save_replay_buffer'] = get_bool("save_replay_buffer", True)
    params['init_entcoef'] = get_input("init_entcoef", 0.1, float)  # DrQv2 tem init_entcoef
    params['image_pad'] = get_input("image_pad", 4, int)
    return OrderedDict([('trainer_type', 'drqv2'), ('hyperparameters', params)])

def get_ppo_et_params(defaults):
    print("\n--- Configuring PPO-ET ---")
    params = OrderedDict()
    params['batch_size'] = get_input("batch_size", defaults.get('batch_size', 128), int)
    params['buffer_size'] = get_input("buffer_size", defaults.get('buffer_size', 10240), int)
    params['learning_rate'] = get_input("learning_rate", defaults.get('learning_rate', 0.0003), float)
    params['beta'] = get_input("beta", 0.01, float)
    params['epsilon'] = get_input("epsilon", 0.2, float)
    params['lambd'] = get_input("lambd", 0.95, float)
    params['num_epoch'] = get_input("num_epoch", 3, int)
    params['learning_rate_schedule'] = get_choice("learning_rate_schedule", ["linear", "constant"], 0)
    params['entropy_temperature'] = get_input("entropy_temperature", 1.0, float)
    params['adaptive_entropy_temperature'] = get_bool("adaptive_entropy_temperature", True)
    
    # Tratar target_entropy com mais cuidado - só adicionar se for um valor numérico válido
    target_entropy_input = get_input("target_entropy (or 'null')", 'null', float)
    if target_entropy_input.lower() != 'none' and target_entropy_input.strip() != '':
        try:
            target_entropy_value = float(target_entropy_input)
            params['target_entropy'] = target_entropy_value
        except ValueError:
            print(f"Warning: Invalid target_entropy value '{target_entropy_input}', skipping this parameter.")
    # Se for 'None' ou vazio, simplesmente não adicionamos o parâmetro ao dicionário
    
    return OrderedDict([('trainer_type', 'ppo_et'), ('hyperparameters', params)])

def get_ppo_ce_params(defaults):
    print("\n--- Configuring PPO-CE ---")
    params = OrderedDict()
    params['batch_size'] = get_input("batch_size", defaults.get('batch_size', 128), int)
    params['buffer_size'] = get_input("buffer_size", defaults.get('buffer_size', 10240), int)
    params['learning_rate'] = get_input("learning_rate", defaults.get('learning_rate', 0.0003), float)
    params['beta'] = get_input("beta", 0.01, float)
    params['epsilon'] = get_input("epsilon", 0.2, float)
    params['lambd'] = get_input("lambd", 0.95, float)
    params['num_epoch'] = get_input("num_epoch", 3, int)
    params['learning_rate_schedule'] = get_choice("learning_rate_schedule", ["linear", "constant"], 0)
    params['curiosity_strength'] = get_input("curiosity_strength", 0.01, float)
    params['curiosity_gamma'] = get_input("curiosity_gamma", 0.99, float)
    params['curiosity_learning_rate'] = get_input("curiosity_learning_rate", 0.0001, float)
    params['curiosity_hidden_units'] = get_input("curiosity_hidden_units", 256, int)
    params['curiosity_num_layers'] = get_input("curiosity_num_layers", 2, int)
    params['imagination_horizon'] = get_input("imagination_horizon", 5, int)
    params['use_imagination_augmented'] = get_bool("use_imagination_augmented", True)
    return OrderedDict([('trainer_type', 'ppo_ce'), ('hyperparameters', params)])

def get_sac_ae_params(defaults):
    print("\n--- Configuring SAC-AE ---")
    params = OrderedDict()
    params['learning_rate'] = get_input("learning_rate", defaults.get('learning_rate', 0.0003), float)
    params['learning_rate_schedule'] = get_choice("learning_rate_schedule", ["linear", "constant"], 0)
    params['batch_size'] = get_input("batch_size", defaults.get('batch_size', 64), int)
    params['buffer_size'] = get_input("buffer_size", defaults.get('buffer_size', 50000), int)
    params['buffer_init_steps'] = get_input("buffer_init_steps", 10000, int)
    params['tau'] = get_input("tau", 0.005, float)
    params['steps_per_update'] = get_input("steps_per_update", 1.0, float)
    params['save_replay_buffer'] = get_bool("save_replay_buffer", True)
    params['init_entcoef'] = get_input("init_entcoef", 0.1, float)  # SAC-AE tem init_entcoef
    params['latent_size'] = get_input("latent_size", 512, int)
    params['ae_learning_rate'] = get_input("ae_learning_rate", 1e-3, float)
    params['ae_hidden_units'] = get_input("ae_hidden_units", 256, int)
    params['ae_num_layers'] = get_input("ae_num_layers", 2, int)
    params['world_model_learning_rate'] = get_input("world_model_learning_rate", 3e-4, float)
    params['world_model_hidden_units'] = get_input("world_model_hidden_units", 256, int)
    params['world_model_num_layers'] = get_input("world_model_num_layers", 2, int)
    params['use_autoencoder'] = get_bool("use_autoencoder", True)
    params['use_world_model'] = get_bool("use_world_model", True)
    params['ae_loss_weight'] = get_input("ae_loss_weight", 1.0, float)
    params['world_model_loss_weight'] = get_input("world_model_loss_weight", 1.0, float)
    params['reconstruction_loss_weight'] = get_input("reconstruction_loss_weight", 1.0, float)
    return OrderedDict([('trainer_type', 'sac_ae'), ('hyperparameters', params)])

# --- Main Generator ---

def create_interactive_rl_yaml():
    config = OrderedDict()
    
    print("\n--- Interactive ML-Agents RL Configuration Generator ---")
    filename = get_input("\nEnter the final output filename", "config.yaml")

    # --- Behaviors Section ---
    behavior_name = get_input("Enter the Behavior Name from your Unity Agent script", "BehaviorName")
    
    base_trainer_settings = OrderedDict()
    print(f"\n--- Base Settings for '{behavior_name}' ---")
    
    all_trainer_keys = ["ppo", "sac", "tdsac", "td3", "tqc", "dcac", "crossq", "drqv2", "ppo_et", "ppo_ce", "sac_ae", "all"]
    trainer_type = get_choice("Choose a base trainer_type (or 'all' to configure multiple)", all_trainer_keys, len(all_trainer_keys) - 1)
    base_trainer_settings['trainer_type'] = trainer_type

    print("\n--- Base Hyperparameters (can be overridden by sub-trainers) ---")
    base_hyperparameters = OrderedDict()
    base_hyperparameters['batch_size'] = get_input("batch_size", 128, int)
    base_hyperparameters['buffer_size'] = get_input("buffer_size", 10240, int)
    base_hyperparameters['learning_rate'] = get_input("learning_rate", 0.0001, float)
    base_hyperparameters['learning_rate_schedule'] = get_choice("learning_rate_schedule", ["linear", "constant"], 0)
    base_trainer_settings['hyperparameters'] = base_hyperparameters

    print("\n--- General Network Settings ---")
    network_settings = get_common_network_settings()
    base_trainer_settings['network_settings'] = network_settings

    print("\n--- General Reward Signals ---")
    base_trainer_settings['reward_signals'] = get_common_reward_signals()

    print("\n--- General Trainer Settings ---")
    common_settings = get_common_trainer_settings()
    for k, v in common_settings.items():
        base_trainer_settings[k] = v

    # --- Sub-trainer configurations if 'all' was chosen ---
    sub_trainers = OrderedDict()
    if trainer_type == 'all':
        available_trainers = {
            "ppo": get_ppo_params,
            "sac": get_sac_params,
            "tdsac": get_tdsac_params,
            "td3": get_td3_params,
            "tqc": get_tqc_params,
            "dcac": get_dcac_params,
            "crossq": get_crossq_params,
            "drqv2": get_drqv2_params,
            "ppo_et": get_ppo_et_params,
            "ppo_ce": get_ppo_ce_params,
            "sac_ae": get_sac_ae_params
        }
        chosen_trainers = get_multiple_choices("Select sub-trainers to configure", list(available_trainers.keys()))
        
        for t_key in chosen_trainers:
            if t_key in available_trainers:
                trainer_config = available_trainers[t_key](base_hyperparameters)
                trainer_config['network_settings'] = network_settings
                trainer_config['reward_signals'] = get_common_reward_signals()
                for k, v in common_settings.items():
                    trainer_config[k] = v
                trainer_config['init_path'] = None
                sub_trainers[t_key] = trainer_config

    # --- Final Assembly ---
    config['behaviors'] = OrderedDict([(behavior_name, base_trainer_settings)])
    for key, val in sub_trainers.items():
        config['behaviors'][behavior_name][key] = val

    # --- Environment Settings ---
    print("\n--- Environment Settings ---")
    env_settings = OrderedDict()
    if trainer_type == 'all' and sub_trainers:
        env_settings['worker_trainer_types'] = list(sub_trainers.keys())
        num_envs_default = len(sub_trainers)
    else:
        num_envs_default = 1
    env_settings['env_path'] = get_input("env_path", "path/to/your/env.x86_64")
    env_settings['env_args'] = get_input("env_args (or None)", "None", str)
    env_settings['base_port'] = get_input("base_port", 5005, int)
    env_settings['num_envs'] = get_input("num_envs", num_envs_default, int)
    env_settings['num_areas'] = get_input("num_areas", 1, int)
    env_settings['timeout_wait'] = get_input("timeout_wait", 60, int)
    env_settings['seed'] = get_input("seed", -1, int)
    env_settings['max_lifetime_restarts'] = get_input("max_lifetime_restarts", 10, int)
    env_settings['restarts_rate_limit_n'] = get_input("restarts_rate_limit_n", 1, int)
    env_settings['restarts_rate_limit_period_s'] = get_input("restarts_rate_limit_period_s", 60, int)
    config['env_settings'] = env_settings

    print("\n--- Environment Parameters (Curriculum) ---")
    env_params = OrderedDict()
    if get_bool("Add environment parameters?", False):
        while True:
            param_name = get_input("Parameter name (or leave blank to finish)")
            if not param_name:
                break
            param_val_str = get_input(f"  Value for '{param_name}'")
            try:
                param_val = float(param_val_str)
            except (ValueError, TypeError):
                param_val = param_val_str
            env_params[param_name] = param_val
    config['environment_parameters'] = env_params

    print("\n--- Engine Settings ---")
    engine_settings = OrderedDict()
    engine_settings['width'] = get_input("width", 250, int)
    engine_settings['height'] = get_input("height", 250, int)
    engine_settings['quality_level'] = get_input("quality_level", 0, int)
    engine_settings['time_scale'] = get_input("time_scale", 200.0, float)
    engine_settings['target_frame_rate'] = get_input("target_frame_rate", 60, int)
    engine_settings['capture_frame_rate'] = get_input("capture_frame_rate", 0, int)
    engine_settings['no_graphics'] = get_bool("no_graphics", True)
    config['engine_settings'] = engine_settings

    print("\n--- Checkpoint Settings ---")
    checkpoint_settings = OrderedDict()
    checkpoint_settings['run_id'] = get_input("run_id", "MyFirstRun")
    checkpoint_settings['load_model'] = get_bool("load_model", False)
    checkpoint_settings['resume'] = get_bool("resume", False)
    checkpoint_settings['force'] = get_bool("force", False)
    checkpoint_settings['train_model'] = get_bool("train_model", True)
    checkpoint_settings['inference'] = get_bool("inference", False)
    checkpoint_settings['results_dir'] = get_input("results_dir", "results")
    config['checkpoint_settings'] = checkpoint_settings

    print("\n--- Torch Settings ---")
    torch_settings = OrderedDict()
    torch_settings['device'] = get_choice("device", ["cuda", "cpu"], 0)
    config['torch_settings'] = torch_settings

    config['debug'] = get_bool("\nEnable debug mode?", False)

    # --- Final Save ---
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            final_dict = odict_to_dict(config)
            yaml.dump(final_dict, f, sort_keys=False, default_flow_style=False, indent=2)
        print(f"\nConfiguration file '{filename}' generated successfully!")
        print(f"You can now run training with: mlagents-learn {filename}")
    except Exception as e:
        print(f"\nError saving file: {e}")

if __name__ == "__main__":
    create_interactive_rl_yaml()