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
    A unified Site Data Provider for all WOFOST versions.

    This class reads parameter definitions and profiles from a YAML configuration file.
    It validates user input against the specific requirements of the selected WOFOST
    model version (profile).

    :param profile: The name of the WOFOST profile to use.
                    Options: 'WOFOST72', 'WOFOST73', 'WOFOST81_Classic', 'WOFOST81_SNOMIN'
    :param config_file: Optional path to the YAML config file. If None, it looks for
                        'wofost_site_parameters.yaml' in the same directory as this script.
    :param kwargs: Keyword arguments containing the site parameter values.
    """

    def __init__(self, profile, **kwargs):
        dict.__init__(self)

        # Load configuration using pkg_resources
        with pkg_resources.files(configs).joinpath("site_params.yaml").open("r") as f:
            full_config = yaml.safe_load(f)

        config = full_config["wofost"]

        if profile not in config["profiles"]:
            raise SiteParameterError(
                f"Unknown profile '{profile}'. Available profiles: {list(config['profiles'].keys())}"
            )

        # Get profile specific settings
        profile_def = config["profiles"][profile]
        self.valid_param_names = set(profile_def["parameters"])
        self.required_params = set(profile_def.get("required", []))
        all_param_defs = config["site_params"]
        self.param_metadata = {
            param: all_param_defs[param] for param in self.valid_param_names
        }

        # 1. Process all valid parameters for this profile
        for par_name in self.valid_param_names:
            par_def = all_param_defs[par_name]

            # Check if provided in kwargs
            if par_name in kwargs:
                value = kwargs.pop(par_name)
            else:
                # Not provided: Check if required or use default
                if par_name in self.required_params:
                    raise SiteParameterError(
                        f"Value for parameter '{par_name}' is required for profile '{profile}'!"
                    )
                value = par_def["default"]

            # Type conversion and Range checking
            if value is not None:
                value = self._convert_and_validate(par_name, value, par_def)

            # Store the value
            self[par_name] = value

        # 2. Check for unknown parameters passed by user (parameters not in the profile)
        if kwargs:
            msg = f"Unknown parameters provided for profile '{profile}': {list(kwargs.keys())}"
            raise SiteParameterError(msg)

    def _convert_and_validate(self, name, value, definition):
        """Helper to convert types and check ranges based on YAML definition."""
        target_type_str = definition["type"]

        # Convert type
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

        # Check Range
        valid_range = definition["range"]

        if target_type_str == "list":
            # For lists, we check if values inside are within range
            min_val, max_val = valid_range
            if not all(min_val <= x <= max_val for x in value):
                raise SiteParameterError(
                    f"Elements in list '{name}' must be between {min_val} and {max_val}"
                )

        elif target_type_str == "int" and valid_range == [0, 1]:
            # Specific handling for flags if strict [0, 1] range provided
            if value not in [0, 1]:
                raise SiteParameterError(f"Parameter '{name}' must be 0 or 1.")

        else:
            # Standard min/max check
            min_val, max_val = valid_range
            if not (min_val <= value <= max_val):
                raise SiteParameterError(
                    f"Value {value} for parameter '{name}' out of range [{min_val}, {max_val}]"
                )

        return value
