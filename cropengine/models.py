"""Module to fetch information about models"""

import os
import yaml
import importlib.resources as pkg_resources
from . import configs

def get_available_models():
    with pkg_resources.files(configs).joinpath("models.yaml").open("r") as f:
        full_config = yaml.safe_load(f)
    
    models_list = []
    for category in full_config.keys():
        models_list.extend(full_config[category]['models'])
    
    return models_list