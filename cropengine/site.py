"""Module to prepare site data"""

import os
import yaml
import importlib.resources as pkg_resources
from . import configs


class SiteParameterError(Exception):
    """Custom exception for site parameter validation errors."""

    pass

class WOFOSTSiteParametersProvider(dict):
    """
    A unified data provider for WOFOST site-specific parameters.
    
    Args:
        profile (str): The name of the WOFOST model version to use.
        (e.g., 'WOFOST72', 'WOFOST73', 'WOFOST81_SNOMIN').
        **kwargs: Site parameters provided as keyword arguments.
        (e.g., WAV=10.0, CO2=400, NH4I=[...]).
        
    Attributes:
        valid_param_names (set): A set of parameter names allowed for the selected profile.
        required_params (set): A set of parameter names that must be provided by the user.
        param_metadata (dict): Metadata (range, type, default) for the valid parameters.
        
    Raises:
        SiteParameterError: If the profile is unknown, if required parameters
                            are missing, or if provided parameters fail validation
                            (incorrect type, out of range, or unknown key).
    """

    def __init__(self, profile, **kwargs):
        dict.__init__(self)

        # Load configuration using pkg_resources (modern approach)
        # Assumes 'configs' is a python package containing 'site_params.yaml'
        with pkg_resources.files(configs).joinpath("site_params.yaml").open("r") as f:
            full_config = yaml.safe_load(f)

        config = full_config["wofost"]

        # Validate Profile
        if profile not in config["profiles"]:
            raise SiteParameterError(
                f"Unknown profile '{profile}'. Available profiles: {list(config['profiles'].keys())}"
            )

        # Load Profile Settings
        profile_def = config["profiles"][profile]
        self.valid_param_names = set(profile_def["parameters"])
        self.required_params = set(profile_def.get("required", []))
        
        all_param_defs = config["site_params"]
        self.param_metadata = {
            param: all_param_defs[param] for param in self.valid_param_names
        }

        # 1. Process and Validate Parameters
        for par_name in self.valid_param_names:
            par_def = all_param_defs[par_name]

            # Determine value: use provided kwarg or fall back to default
            if par_name in kwargs:
                value = kwargs.pop(par_name)
            else:
                if par_name in self.required_params:
                    raise SiteParameterError(
                        f"Value for parameter '{par_name}' is required for profile '{profile}'!"
                    )
                value = par_def["default"]

            # Convert types and check valid ranges
            if value is not None:
                value = self._convert_and_validate(par_name, value, par_def)

            # Store validated value in the dictionary
            self[par_name] = value

        # 2. Check for Unknown Parameters
        # Any keys remaining in kwargs are not defined in the profile configuration
        if kwargs:
            msg = f"Unknown parameters provided for profile '{profile}': {list(kwargs.keys())}"
            raise SiteParameterError(msg)

    def _convert_and_validate(self, name, value, definition):
        """
        Internal helper to cast types and validate ranges.

        Args:
            name (str): The name of the parameter.
            value (any): The value to validate.
            definition (dict): The metadata dictionary containing 'type' and 'range'.

        Returns:
            The value cast to the correct type (int, float, list).

        Raises:
            SiteParameterError: If type casting fails or value is out of bounds.
        """
        target_type_str = definition["type"]

        # 1. Type Casting
        try:
            if target_type_str == "int":
                value = int(value)
            elif target_type_str == "float":
                value = float(value)
            elif target_type_str == "list":
                if not isinstance(value, list):
                    raise ValueError
        except (ValueError, TypeError):
            raise SiteParameterError(
                f"Parameter '{name}' must be of type {target_type_str}, got {type(value)}"
            )

        # 2. Range Checking
        valid_range = definition["range"]

        if target_type_str == "list":
            # For lists, validate every element against the range
            min_val, max_val = valid_range
            if not all(min_val <= x <= max_val for x in value):
                raise SiteParameterError(
                    f"Elements in list '{name}' must be between {min_val} and {max_val}"
                )

        elif target_type_str == "int" and valid_range == [0, 1]:
            # Specific handling for binary flags (0 or 1)
            if value not in [0, 1]:
                raise SiteParameterError(f"Parameter '{name}' must be 0 or 1.")

        else:
            # Standard numeric range check [min, max]
            min_val, max_val = valid_range
            if not (min_val <= value <= max_val):
                raise SiteParameterError(
                    f"Value {value} for parameter '{name}' out of range [{min_val}, {max_val}]"
                )

        return value