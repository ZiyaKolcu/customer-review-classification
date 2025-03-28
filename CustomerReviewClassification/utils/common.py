import os
import sys
import yaml
import joblib
import json
from CustomerReviewClassification.exception.exception import CustomException
from CustomerReviewClassification.logging.logger import logging
from pathlib import Path
from box import ConfigBox
from box.exceptions import BoxValueError


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)

    except BoxValueError:
        raise ValueError("yaml file is empty")

    except Exception as e:
        raise CustomException(e, sys)


def create_directories(path_to_directories: list, verbose=True):
    """create list of directories
    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"created directory at: {path}")


def save_bin(data, file_path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file_obj:
        joblib.dump(data, file_obj)

    logging.info(f"binary file saved at: {file_path}")


def load_bin(path: Path):
    """load binary data
    Args:
        path (Path): path to binary file
    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logging.info(f"binary file loaded from: {path}")
    return data


def save_json(path: Path, data: dict):
    """save json data
    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logging.info(f"json file saved at: {path}")
