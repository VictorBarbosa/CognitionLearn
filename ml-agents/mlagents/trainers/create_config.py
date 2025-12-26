import yaml

def get_input(prompt, default=None):
    """Helper to get user input with a default value."""
    default_text = f" (default: {default})" if default is not None else ""
    response = input(f"{prompt}{default_text}: ")
    return response if response else default

def get_list_input(prompt):
    """Helper to get a comma-separated list from user."""
    response = input(f"{prompt} (comma-separated): ")
    # Return a string that looks like a YAML list
    return str([item.strip() for item in response.split(',') if item.strip()])

def get_choice_input(prompt, options, default_index=0):
    """Helper to get a choice from a list of options."""
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

def get_multiple_choices_input(prompt, options):
    """Helper to get multiple choices from a list of options."""
    print(prompt)
    for i, option in enumerate(options, 1):
        print(f"  {i}: {option}")
    
    while True:
        response = input(f"Choose one or more options (comma-separated, e.g., 1,3): ")
        if not response:
            print("No option selected. Please choose at least one.")
            continue
        try:
            choices_str = response.split(',')
            selected_options = []
            valid = True
            for choice_str in choices_str:
                choice_index = int(choice_str.strip()) - 1
                if 0 <= choice_index < len(options):
                    if options[choice_index] not in selected_options:
                         selected_options.append(options[choice_index])
                else:
                    print(f"Invalid option: {choice_str}. Try again.")
                    valid = False
                    break
            if valid and selected_options:
                return selected_options
        except ValueError:
            print("Please enter comma-separated numbers.")

def get_bool_input(prompt, default=True):
    """Helper to get a boolean (Yes/No) from the user."""
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

YAML_TEMPLATE = """# Complete configuration for standalone supervised training

# Training data settings
data:
  # Path to the CSV file with training data
  csv_path: "{csv_path}"
  
  # Observation columns in the CSV (leave empty to use all except action columns)
  observation_columns: {observation_columns}
  
  # Action columns in the CSV (required)
  action_columns: {action_columns}
  
  # Fraction of the data to be used for validation
  validation_split: 0.2
  
  # Whether the data should be shuffled
  shuffle: {data_shuffle}
  
  # Noise level for data augmentation
  augment_noise: 0.01

# Model settings
model:
  # Algorithm(s) to be used for architecture compatibility (ppo, sac, tdsac)
  # Can be a list, e.g.: [ppo, sac]
  algorithm: {algorithm}
  
  # Neural network settings
  network_settings:
    # Number of hidden units in each layer
    hidden_units: {hidden_units}
    
    # Number of hidden layers
    num_layers: {num_layers}
    
    # Whether observations should be normalized
    normalize: {normalize}
    
    # Memory size for recurrent networks (0 for disabled)
    memory_size: {memory_size}
    
    # Whether to use recurrent networks
    use_recurrent: {use_recurrent}

# Training settings
training:
  # Number of training epochs
  epochs: {epochs}
  
  # Batch size
  batch_size: {batch_size}
  
  # Learning rate
  learning_rate: {learning_rate}
  
  # Fraction of the data to be used for validation
  validation_split: 0.2
  
  # Whether the data should be shuffled
  shuffle: {training_shuffle}
  
  # Noise level for data augmentation
  augment_noise: 0.01
  
  # Whether to use early stopping
  early_stopping: {early_stopping}
  
  # Number of epochs without improvement before stopping
  patience: 10
  
  # Minimum change to be considered an improvement
  min_delta: 0.001
  
  # Dropout rate for regularization
  dropout_rate: 0.1
  
  # Weight decay for L2 regularization
  weight_decay: 1e-4
  
  # Patience for learning rate reduction
  lr_patience: 5

# Output settings
output:
  # Directory to save the results
  dir: "{output_dir}"
  
  # Epoch interval to save checkpoints
  checkpoint_interval: {checkpoint_interval}
  
  # Whether to export to ONNX
  export_onnx: {export_onnx}
  
  # Whether to export to .pt
  export_pt: {export_pt}

# Additional settings
additional_settings:
  # Device to be used (cuda or cpu)
  device: "{device}"
  
  # Seed for reproducibility
  seed: 42
  
  # Whether to show detailed information
  verbose: {verbose}
"""

def create_interactive_yaml():
    """Main function to generate YAML interactively based on the new format."""
    print("\n--- Interactive Supervised Configuration Generator ---\n")
    
    filename = get_input("Output file name", "config_supervised.yaml")
    
    print("\n--- Data Settings ---")
    csv_path = get_input("Path to the CSV file", "./data/training_data.csv")
    obs_cols = get_list_input("Observation column names")
    action_cols = get_list_input("Action column names")
    data_shuffle = get_bool_input("Shuffle data?", default=True)
    
    print("\n--- Model Settings ---")
    all_algorithms = ["ppo", "sac", "tdsac", "td3", "tqc", "dcac", "crossq", "drqv2", "ppo_et", "ppo_ce", "sac_ae"]
    
    # Check if user wants to select all algorithms
    if get_bool_input("Use all available algorithms?", default=False):
        algorithms = all_algorithms
        print(f"Selected all algorithms: {algorithms}")
    else:
        algorithms = get_multiple_choices_input("Target algorithm(s) for compatibility", all_algorithms)
    hidden_units = int(get_input("Hidden units per layer", 256))
    num_layers = int(get_input("Number of hidden layers", 2))
    normalize = get_bool_input("Normalize observations?", default=True)
    use_recurrent = get_bool_input("Use recurrent network (RNN)?", default=False)
    memory_size = 0
    if use_recurrent:
        memory_size = int(get_input("Memory size (RNN)", 128))

    print("\n--- Training Settings ---")
    epochs = int(get_input("Number of epochs", 100))
    batch_size = int(get_input("Batch size", 128))
    learning_rate = float(get_input("Learning rate", 0.0003))
    early_stopping = get_bool_input("Use early stopping?", default=True)
    
    print("\n--- Output Settings ---")
    output_dir = get_input("Output directory", "./results")
    checkpoint_interval = int(get_input("Checkpoint interval (epochs)", 1000))
    export_onnx = get_bool_input("Export model to ONNX?", default=True)
    export_pt = get_bool_input("Export model to .pt (PyTorch)?", default=True)

    print("\n--- Additional Settings ---")
    device = get_choice_input("Training device", ["cpu", "cuda"], default_index=0)
    verbose = get_bool_input("Enable verbose mode?", default=True)

    # Format the YAML template with user inputs
    # Convert booleans to lowercase 'true'/'false' for YAML
    final_yaml = YAML_TEMPLATE.format(
        csv_path=csv_path,
        observation_columns=obs_cols,
        action_columns=action_cols,
        data_shuffle=str(data_shuffle).lower(),
        algorithm=str(algorithms),
        hidden_units=hidden_units,
        num_layers=num_layers,
        normalize=str(normalize).lower(),
        memory_size=memory_size,
        use_recurrent=str(use_recurrent).lower(),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        training_shuffle=str(data_shuffle).lower(),
        early_stopping=str(early_stopping).lower(),
        output_dir=output_dir,
        checkpoint_interval=checkpoint_interval,
        export_onnx=str(export_onnx).lower(),
        export_pt=str(export_pt).lower(),
        device=device,
        verbose=str(verbose).lower()
    )

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(final_yaml.strip())
        print(f"\nConfiguration file '{filename}' generated successfully!\nHow to run: mlagents-supervised --config {filename} \n               \n               ")
    except Exception as e:
        print(f"\nError saving file: {e}")

if __name__ == "__main__":
    create_interactive_yaml()
