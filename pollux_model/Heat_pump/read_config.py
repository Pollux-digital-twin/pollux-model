import yaml


def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def write_config(file_path, config):
    with open(file_path, 'w') as file:
        yaml.safe_dump(config, file)


def display_config(config):
    print("Current Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")


def update_config(config):
    for key in config.keys():
        new_value = input(f"Enter new value for {key} "
                          f"(leave blank to keep current value '{config[key]}'): ")
        if new_value:
            config[key] = new_value
