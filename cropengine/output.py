"""Module to get output variables"""

import yaml
import importlib.resources as pkg_resources
from . import configs


def get_output_variables(model: str) -> list[dict]:
    """
    Retrieve output variable definitions and metadata for a given simulation model.

    Args:
        model (str): Name of the simulation model (e.g., a WOFOST variant).

    Returns:
        list[dict]: A list of dictionaries, each describing an output variable.
        Each dictionary contains the following keys:
    """

    try:
        with pkg_resources.files(configs).joinpath("output.yaml").open("r") as f:
            full_config = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load output.yaml: {e}")

    if model.lower().startswith("wofost"):
        config = full_config["wofost"]

    # Get the output variable
    if model in config["model_mapping"]:
        profile_name = config["model_mapping"][model]
        variable_names = config["profiles"][profile_name]["variables"]

        variable_defs = []

        # Build the Metadata List
        for var in variable_names:
            meta = {
                "variable": var,
                "description": config["output_vars"][var]["description"],
                "unit": config["output_vars"][var]["unit"],
                "type": config["output_vars"][var]["type"],
            }
            variable_defs.append(meta)

    return variable_defs
