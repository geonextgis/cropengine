"""Main cropengine module."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# CropEngine Imports
from cropengine.models import get_available_models, get_model_class
from cropengine.crop import get_available_crops, get_available_crop_varieties
from cropengine.weather import GEEWeatherDataProvider
from cropengine.soil import (
    IsricSoilDataProvider,
    GEEIsricSoilDataProvider,
    WOFOSTSoilParameterProvider,
)
from cropengine.site import WOFOSTSiteParametersProvider
from cropengine.crop import WOFOSTCropParametersProvider
from cropengine.agromanagement import (
    WOFOSTAgroEventBuilder,
    WOFOSTAgroManagementProvider,
)

# PCSE Imports
from pcse.base import ParameterProvider
from pcse.input import YAMLAgroManagementReader, ExcelWeatherDataProvider
import ast


class WOFOSTCropSimulationRunner:
    def __init__(self, workspace_dir="workspace"):
        """
        Initialize the runner.
        Calls update_workspace immediately to set up file paths.
        """
        self.update_workspace(workspace_dir)

    # =========================================================================
    # 0. WORKSPACE & PATH MANAGEMENT
    # =========================================================================
    def update_workspace(self, new_dir):
        """
        Updates the workspace directory and refreshes all file paths.
        Call this method when switching to a new simulation point in a loop.
        """
        self.workspace_dir = new_dir
        os.makedirs(self.workspace_dir, exist_ok=True)

        self.files = {
            "weather": os.path.join(self.workspace_dir, "meteo.xlsx"),
            "soil": os.path.join(self.workspace_dir, "soil.csv"),
            "soil_params": os.path.join(self.workspace_dir, "params_soil.csv"),
            "site_params": os.path.join(self.workspace_dir, "params_site.csv"),
            "crop_params": os.path.join(self.workspace_dir, "params_crop.csv"),
            "agro": os.path.join(self.workspace_dir, "agro.yaml"),
            "output": os.path.join(self.workspace_dir, "simulation_results.csv"),
        }

    # =========================================================================
    # 1. UI HELPERS (Dropdowns)
    # =========================================================================
    def get_model_options(self):
        """Returns list of available models."""
        return get_available_models()

    def get_crop_options(self, model_name):
        """Returns list of crops for the selected model."""
        return get_available_crops(model_name)

    def get_variety_options(self, model_name, crop_name):
        """Returns list of varieties for the selected crop."""
        return get_available_crop_varieties(model_name, crop_name)

    def get_crop_start_end_options(self):
        """Returns dict of crop start and crop end type available in WOFOST."""
        return {
            "crop_start_type": ["sowing", "emergence"],
            "crop_end_type": ["maturity", "harvest", "earliest"],
        }

    # =========================================================================
    # 2. DATA PREPARATION (I/O Bound)
    #    Downloads data and writes config files. Run this BEFORE the simulation.
    # =========================================================================
    def prepare_system(
        self,
        latitude,
        longitude,
        campaign_start,
        campaign_end,
        crop_start,
        crop_end,
        model_name,
        crop_name,
        variety_name,
        crop_start_type="emergence",
        crop_end_type="harvest",
        max_duration=300,
        timed_events=None,
        state_events=None,
        force_update=False,
        **site_kwargs,
    ):
        """
        Prepares the workspace:
        1. Downloads Weather/Soil if missing.
        2. Generates and saves Parameter CSVs (Soil, Site, Crop).
        3. Generates and saves Agromanagement YAML.
        """
        print(f"[PREP] Preparing workspace: {self.workspace_dir}")

        # A. Weather & Soil (Download/Cache)
        self._ensure_weather(
            latitude, longitude, campaign_start, campaign_end, force_update
        )
        soil_raw_path = self._ensure_soil(latitude, longitude, force_update)

        # B. Parameter Generation (Save to CSV)
        # 1. Soil
        soil_df = pd.read_csv(soil_raw_path)
        self._save_params(WOFOSTSoilParameterProvider(soil_df), "soil_params")

        # 2. Site
        site_provider = WOFOSTSiteParametersProvider(model_name, **site_kwargs)
        _ = site_provider.get_params()
        self._save_params(site_provider, "site_params")

        # 3. Crop
        self._save_params(
            WOFOSTCropParametersProvider(crop_name, variety_name), "crop_params"
        )

        # C. Agromanagement Generation (Save to YAML)
        self._build_agromanagement(
            campaign_start,
            campaign_end,
            crop_start,
            crop_end,
            crop_name,
            variety_name,
            crop_start_type,
            crop_end_type,
            max_duration,
            timed_events,
            state_events,
        )

        print("[PREP] System Ready.")

    # =========================================================================
    # 3. SIMULATION EXECUTION
    #    Pure logic. Reads prepared files and runs math.
    # =========================================================================
    def run_simulation(
        self,
        model_name,
        soil_params_df=None,
        site_params_df=None,
        crop_params_df=None,
        agro_file_path=None,
    ):
        """
        Runs the simulation.

        Args:
            model_name (str): Model ID.
            soil_params_df (pd.DataFrame, optional): Override soil params.
            site_params_df (pd.DataFrame, optional): Override site params.
            crop_params_df (pd.DataFrame, optional): Override crop params.
            agro_file (str, optional): Path to a user-provided YAML file.
                                       If None, uses the workspace default.
        """
        print(f"[RUN] Initializing {model_name} in {self.workspace_dir}...")

        # 1. Resolve Agromanagement File
        if agro_file_path:
            if not os.path.exists(agro_file_path):
                raise FileNotFoundError(
                    f"Custom agromanagement file not found: {agro_file_path}"
                )
            agro_path = agro_file_path
            print(f"[RUN] Using custom agromanagement: {agro_path}")
        else:
            if not os.path.exists(self.files["agro"]):
                raise FileNotFoundError(
                    "Default agromanagement file missing. Run 'prepare_system' or provide 'custom_agro_file'."
                )
            agro_path = self.files["agro"]

        # 1. Load Weather (Must exist)
        if not os.path.exists(self.files["weather"]) or not os.path.exists(
            self.files["agro"]
        ):
            raise FileNotFoundError(
                "Missing weather or agro files. Run 'prepare_system' first."
            )

        weather_provider = ExcelWeatherDataProvider(self.files["weather"])
        agromanagement = YAMLAgroManagementReader(agro_path)

        # 2. Load Parameters (Priority: DF Input -> CSV File -> Error)
        soil_dict = self._load_param_dict(soil_params_df, "soil_params", "value")
        site_dict = self._load_param_dict(site_params_df, "site_params", "value")
        crop_dict = self._load_param_dict(crop_params_df, "crop_params", "default")
        parameters = ParameterProvider(
            cropdata=crop_dict, soildata=soil_dict, sitedata=site_dict
        )

        # 3. Instantiate & Run
        try:
            ModelClass = get_model_class(model_name)
            wofsim = ModelClass(parameters, weather_provider, agromanagement)
            wofsim.run_till_terminate()

            # 4. Save Output
            df_results = pd.DataFrame(wofsim.get_output())
            df_results.to_csv(self.files["output"], index=False)
            return df_results

        except Exception as e:
            raise RuntimeError(f"Simulation Failed: {e}")

    # =========================================================================
    # 4. INTERNAL HELPERS
    # =========================================================================
    def _save_params(self, provider, file_key):
        """Helper to extract metadata from a provider and save to CSV."""
        try:
            _ = provider.get_params()
            param_metadata = provider.param_metadata
        except:
            param_metadata = provider.param_metadata

        df = pd.DataFrame(param_metadata).sort_values(by="parameter")
        df.to_csv(self.files[file_key], index=False)

    def _load_param_dict(self, df_override, file_key, value_col):
        """
        Helper to load parameters from DF override or Disk.
        Includes robust type conversion (String -> Float/List) to fix TypeErrors.
        """
        # 1. Determine Source DataFrame
        if df_override is not None:
            df = df_override
        elif os.path.exists(self.files[file_key]):
            df = pd.read_csv(self.files[file_key])
        else:
            raise FileNotFoundError(f"Missing parameter file: {self.files[file_key]}")

        # 2. Determine Column Name
        col = value_col if value_col in df.columns else "default"

        # 3. Build Dictionary with Type Conversion
        params = {}

        # Iterate manually to handle mixed types (floats vs lists) safely
        for _, row in df.iterrows():
            key = row["parameter"]
            raw_val = row[col]

            # Skip if strict value required and missing
            if col == "value" and pd.isna(raw_val):
                continue

            # If value is already a float/int/list (from DataFrame override), use it directly
            if not isinstance(raw_val, str):
                params[key] = raw_val
                continue

            # TYPE CONVERSION LOGIC (String -> Number/List)
            raw_val = raw_val.strip()
            final_val = raw_val  # Default to original string if parsing fails

            # A. Try converting to List (e.g., "[0.0, 1.5]")
            if raw_val.startswith("[") and raw_val.endswith("]"):
                try:
                    final_val = ast.literal_eval(raw_val)
                except (ValueError, SyntaxError):
                    pass

            # B. Try converting to Number (e.g., "10.0")
            else:
                try:
                    final_val = float(raw_val)
                    if final_val.is_integer():
                        final_val = int(final_val)
                except ValueError:
                    pass

            params[key] = final_val

        return params

    def _ensure_weather(self, lat, lon, start, end, force):
        if not force and os.path.exists(self.files["weather"]):
            return

        d_end = (
            datetime.strptime(end, "%Y-%m-%d").date() + timedelta(days=1)
        ).strftime("%Y-%m-%d")
        meteo = GEEWeatherDataProvider(
            start_date=start,
            end_date=d_end,
            latitude=lat,
            longitude=lon,
            filepath=self.files["weather"],
        )
        meteo.save_weather_excel()

    def _ensure_soil(self, lat, lon, force):
        if not force and os.path.exists(self.files["soil"]):
            return self.files["soil"]

        soil = GEEIsricSoilDataProvider(
            latitude=lat, longitude=lon, depths=["0-5cm"], filepath=self.files["soil"]
        )
        soil.get_data()
        return self.files["soil"]

    def _build_agromanagement(
        self,
        c_start,
        c_end,
        start,
        end,
        crop,
        var,
        st_type,
        end_type,
        dur,
        timed_events,
        state_events,
    ):
        agro = WOFOSTAgroManagementProvider()
        agro.add_campaign(
            c_start,
            c_end,
            crop,
            var,
            start,
            end,
            st_type,
            end_type,
            dur,
            timed_events,
            state_events,
        )
        agro.add_trailing_empty_campaign()
        agro.save_to_yaml(self.files["agro"])
