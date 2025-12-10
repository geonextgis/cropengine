"""Module to prepare site data"""

import os
import yaml
import importlib.resources as pkg_resources
from . import configs
import warnings

class SiteParameterError(Exception):
    """Custom exception for site parameter validation errors."""

    pass


class WOFOSTSiteParametersProvider:
    """
    A unified data provider for WOFOST site-specific parameters.

    Args:
        model (str): The name of the WOFOST model version to use.
        **kwargs: Site parameters provided as keyword arguments.
    """
    EMERGENCY_DEFAULTS = {
        "WAV": 10.0,         
        "CO2": 360.0,       
        "NAVAILI": 0.0,      
        "NH4I": [0.05],      
        "NO3I": [0.05] 
    }

    def __init__(self, model, **kwargs):
        self.model = model
        self.raw_kwargs = kwargs
        self.param_metadata = []
        self.required_params = set()
        self.valid_param_names = set()

        # 1. Load configuration
        try:
            with pkg_resources.files(configs).joinpath("site_params.yaml").open(
                "r"
            ) as f:
                self.full_config = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load site_params.yaml: {e}")

        config = self.full_config["wofost"]

        # 2. Validate Model and Prepare Metadata
        if self.model in config["model_mapping"]:

            profile_name = config["model_mapping"][self.model]
            profile_def = config["profiles"][profile_name]

            self.valid_param_names = set(profile_def["parameters"])
            self.required_params = set(profile_def.get("required", []))

            all_param_defs = config["site_params"]

            # Build the Metadata List
            for param in self.valid_param_names:
                if param in all_param_defs:
                    meta = {
                        "parameter": param,
                        "required": (param in self.required_params),
                    }
                    meta.update(all_param_defs[param].copy())
                    self.param_metadata.append(meta)

    def get_params(self):
        """
        Validates inputs against the prepared metadata, applies defaults,
        and returns the final parameter dictionary.
        """
        # 1. Re-validate Model Profile
        if not self.param_metadata:
            valid_models = list(self.full_config["wofost"]["model_mapping"].keys())
            raise SiteParameterError(
                f"Unknown or unconfigured model '{self.model}'. Available models: {valid_models}"
            )

        validated_params = {}

        # 2. Process Parameters (Iterating over the LIST now)
        for meta in self.param_metadata:
            par_name = meta["parameter"]
            is_required = meta["required"]

            # Determine value: use provided kwarg or fall back to default
            if par_name in self.raw_kwargs:
                value = self.raw_kwargs[par_name]
            else:
                value = meta['default']
                
                if value is None:
                    if par_name in self.EMERGENCY_DEFAULTS:
                        value = self.EMERGENCY_DEFAULTS[par_name]
                        warnings.warn(
                            f"[SiteParams] Parameter '{par_name}' was missing and has no YAML default. "
                            f"Using fallback value: {value}"
                        )
  
                if is_required:
                    warnings.warn(
                        f"[SiteParams] Required parameter '{par_name}' was not provided. "
                        f"Using default value: {value}"
                    )
                
            # Convert types and check valid ranges
            if value is not None:
                value = self._convert_and_validate(par_name, value, meta)
                
            meta['value'] = value
            validated_params[par_name] = value

        # 3. Check for Unknown Parameters provided by user
        unknown_keys = [
            k for k in self.raw_kwargs.keys() if k not in self.valid_param_names
        ]
        if unknown_keys:
            raise SiteParameterError(
                f"Unknown parameters provided for profile '{self.model}': {unknown_keys}"
            )

        return validated_params

    def _convert_and_validate(self, name, value, definition):
        """
        Internal helper to cast types and validate ranges.
        """
        target_type_str = definition["type"]

        # Type Casting
        try:
            if target_type_str == "int":
                value = int(value)
            elif target_type_str == "float":
                value = float(value)
            elif target_type_str == "list":
                if not isinstance(value, list):
                    if isinstance(value, str) and "," in value:
                        value = [float(x.strip()) for x in value.split(",")]
                    else:
                        raise ValueError
        except (ValueError, TypeError):
            raise SiteParameterError(
                f"Parameter '{name}' must be of type {target_type_str}, got {type(value)}"
            )

        # Range Checking
        valid_range = definition["range"]

        if target_type_str == "list":
            min_val, max_val = valid_range
            if not all(min_val <= x <= max_val for x in value):
                raise SiteParameterError(
                    f"Elements in list '{name}' must be between {min_val} and {max_val}"
                )

        elif target_type_str == "int" and valid_range == [0, 1]:
            if value not in [0, 1]:
                raise SiteParameterError(f"Parameter '{name}' must be 0 or 1.")

        else:
            min_val, max_val = valid_range
            if not (min_val <= value <= max_val):
                raise SiteParameterError(
                    f"Value {value} for parameter '{name}' out of range [{min_val}, {max_val}]"
                )

        return value
