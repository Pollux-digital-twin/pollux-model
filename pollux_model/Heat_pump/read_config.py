import os
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
        new_value = input(f"Enter new value for {key} (leave blank to keep current value '{config[key]}'): ")
        if new_value:
            config[key] = new_value


def main():
    config_file_path = r'C:\Users\ntagkrasd\PycharmProjects\heat_pump_model\dagi_case\heat_pump_model_inputs.yml'

    # Read the current configuration
    config = read_config(config_file_path)

    # Display the current configuration
    display_config(config)

    # Prompt the user to update the configuration
    update_config(config)

    # Write the updated configuration back to the file
    write_config(config_file_path, config)

    print("Configuration updated successfully.")


if __name__ == "__main__":
    main()

