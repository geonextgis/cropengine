"""Module to prepare crop parameters"""

import yaml
from pcse.input import YAMLCropDataProvider
import importlib.resources as pkg_resources
from .configs import wofost_crop_params


class WOFOSTCropParametersProvider(YAMLCropDataProvider):
    """
    A data provider for WOFOST crop parameters. This class extends the standard YAMLCropDataProvider to automatically
    locate and load crop-specific parameter files.

    Args:
        crop_name (str): The name of the crop (e.g., 'wheat', 'maize').
        variety_name (str): The specific variety of the crop (e.g., 'Winter_wheat_101').
    """

    def __init__(self, crop_name: str, variety_name: str):
        # 1. Get the directory path directly from the module
        config_path = list(wofost_crop_params.__path__)[0]

        # 2. Initialize the parent class with the string path
        super().__init__(fpath=str(config_path))

        # 3. Set the active crop
        self.set_active_crop(crop_name, variety_name)
        self.crop_name = crop_name
        self.variety_name = variety_name
        self.param_metadata = self._get_param_metadata()

    def _get_param_metadata(self) -> list[dict]:
        """
        Retrieves metadata for the current crop variety from the YAML configuration.
        """
        with pkg_resources.files(wofost_crop_params).joinpath(
            f"{self.crop_name}.yaml"
        ).open("r") as f:
            crop_config = yaml.safe_load(f)
            crop_variety_config = crop_config["CropParameters"]["Varieties"][
                self.variety_name
            ]

            param_metadata = []
            for param, info in crop_variety_config.items():
                try:
                    param_dict = {
                        "parameter": param,
                        "description": info[1],
                        "unit": info[-1][0] if len(info[-1]) == 1 else info[-1],
                        "default": info[0],
                    }
                    param_metadata.append(param_dict)
                except (IndexError, TypeError, KeyError):
                    continue

            return param_metadata


def get_available_crops(model: str) -> list[str]:
    """
    Retrieves a list of supported crops for a specific model type.

    Args:
        model (str): The name of the simulation model (e.g., "Wofost72_PP").

    Returns:
        list[str]: A list of available crop names.
    """
    if model.startswith("Wofost"):
        with pkg_resources.files(wofost_crop_params).joinpath("crops.yaml").open(
            "r"
        ) as f:
            available_crops = yaml.safe_load(f)["available_crops"]

        return available_crops

    else:
        return []


def get_available_crop_varieties(model: str, crop: str) -> dict | None:
    """
    Retrieves available varieties and their metadata for a specific crop.

    Args:
        model (str): The name of the simulation model.
        crop (str): The name of the crop (e.g., 'wheat').

    Returns:
        dict | None: A dictionary where keys are variety identifiers and values
        are metadata strings (descriptions). Returns None if the model is not supported.
    """
    if model.startswith("Wofost"):
        with pkg_resources.files(wofost_crop_params).joinpath(f"{crop}.yaml").open(
            "r"
        ) as f:
            crop_config = yaml.safe_load(f)

            all_crop_varieties = crop_config["CropParameters"]["Varieties"].keys()
            crop_varieties = {}

            for v in all_crop_varieties:
                # Extract the descriptive metadata for the variety
                meta = crop_config["CropParameters"]["Varieties"][v]["Metadata"]
                crop_varieties[v] = meta

        return crop_varieties

    else:
        return None
