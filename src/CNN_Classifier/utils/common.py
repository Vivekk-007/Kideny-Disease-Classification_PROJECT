import os
import yaml
import json
from box import ConfigBox
from pathlib import Path
from typing import Any
from CNN_Classifier import logger


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads yaml file and returns ConfigBox
    
    Args:
        path_to_yaml (Path): path to yaml file
    
    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except Exception as e:
        logger.error(f"Error reading yaml file: {e}")
        raise e


def create_directories(path_to_directories: list, verbose: bool = True):
    """
    Create list of directories
    
    Args:
        path_to_directories (list): list of path of directories
        verbose (bool, optional): ignore if multiple dirs is to be created. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


def save_json(path: Path, data: dict):
    """
    Save json data
    
    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"json file saved at: {path}")


def load_json(path: Path) -> ConfigBox:
    """
    Load json files data
    
    Args:
        path (Path): path to json file
    
    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)
    logger.info(f"json file loaded successfully from: {path}")
    return ConfigBox(content)