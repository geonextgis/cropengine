"""Module to fetch information about models"""

import os
import yaml
import importlib.resources as pkg_resources
from . import configs
import pcse
from pcse.models import (
    # WOFOST 7.2
    Wofost72_Phenology,
    Wofost72_PP,
    Wofost72_WLP_CWB,
    # WOFOST 7.3
    Wofost73_PP,
    Wofost73_WLP_CWB,
    Wofost73_WLP_MLWB,
    # WOFOST 8.1
    Wofost81_PP,
    Wofost81_WLP_CWB,
    Wofost81_WLP_MLWB,
    Wofost81_NWLP_CWB_CNB,
    Wofost81_NWLP_MLWB_CNB,
    Wofost81_NWLP_MLWB_SNOMIN,
    # # LINGRA (Grassland)
    # Lingra10_PP,
    # Lingra10_WLP_CWB,
    # Lingra10_NWLP_CWB_CNB,
    # # LINTUL
    # Lintul3, # Maps to Lintul10_NWLP_CWB_CNB
    # # OTHER
    # Alcepas_PP, # Maps to Alcepas10_PP (Check your PCSE version, usually named Alcepas_PP)
    # Fao_WRSI # Maps to FAO_WRSI10_WLP_CWB
)

# Dictionary mapping User String ID -> Actual Python Class
MODEL_CLASS_MAPPING = {
    # --- WOFOST 7.2 ---
    "Wofost72_Phenology": Wofost72_Phenology,
    "Wofost72_PP": Wofost72_PP,
    "Wofost72_WLP_CWB": Wofost72_WLP_CWB,
    # --- WOFOST 7.3 ---
    "Wofost73_PP": Wofost73_PP,
    "Wofost73_WLP_CWB": Wofost73_WLP_CWB,
    "Wofost73_WLP_MLWB": Wofost73_WLP_MLWB,
    # --- WOFOST 8.1 ---
    "Wofost81_PP": Wofost81_PP,
    "Wofost81_WLP_CWB": Wofost81_WLP_CWB,
    "Wofost81_WLP_MLWB": Wofost81_WLP_MLWB,
    "Wofost81_NWLP_CWB_CNB": Wofost81_NWLP_CWB_CNB,
    "Wofost81_NWLP_MLWB_CNB": Wofost81_NWLP_MLWB_CNB,
    "Wofost81_NWLP_MLWB_SNOMIN": Wofost81_NWLP_MLWB_SNOMIN,
    # # --- LINGRA ---
    # "Lingra10_PP": Lingra10_PP,
    # "Lingra10_WLP_CWB": Lingra10_WLP_CWB,
    # "Lingra10_NWLP_CWB_CNB": Lingra10_NWLP_CWB_CNB,
    # # --- LINTUL ---
    # "Lintul10_NWLP_CWB_CNB": Lintul3,
    # # --- SPECIALTY ---
    # "Alcepas10_PP": Alcepas_PP,
    # "FAO_WRSI10_WLP_CWB": Fao_WRSI
}


def get_available_models():
    with pkg_resources.files(configs).joinpath("models.yaml").open("r") as f:
        full_config = yaml.safe_load(f)

    models_list = []
    for category in full_config.keys():
        models_list.extend(full_config[category]["models"])

    return models_list


def get_model_class(model_id):
    """
    Returns the model class based on the string ID.
    Raises ValueError if the model ID is not found.
    """
    if model_id in MODEL_CLASS_MAPPING:
        return MODEL_CLASS_MAPPING[model_id]
    else:
        valid_keys = list(MODEL_CLASS_MAPPING.keys())
        raise ValueError(
            f"Model ID '{model_id}' not found. Available models: {valid_keys}"
        )
