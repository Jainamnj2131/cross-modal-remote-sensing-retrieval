import os
import yaml


def load_config(config_path="configs/config.yaml"):
    """
    Loads project YAML configuration file into a Python dictionary.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration settings dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found at: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
