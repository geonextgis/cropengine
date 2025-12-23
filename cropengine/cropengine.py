"""Main cropengine module."""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import concurrent
import multiprocessing
from tqdm import tqdm

# CropEngine Imports
from cropengine.models import get_available_models, get_model_class
from cropengine.crop import get_available_crops, get_available_crop_varieties
from cropengine.output import get_output_variables
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


class WOFOSTOptionsMixin:
    """
    Mixin class that provides helper methods for UI dropdowns.
    Inherit from this to give your class access to model/crop options.
    """
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
            'crop_start_type': ['sowing', 'emergence'],
            'crop_end_type': ['maturity', 'harvest', 'earliest'] 
        }
        
    def get_output_variables(self, model_name):
        """Returns list of output variables for a given simulation model."""
        return get_output_variables(model_name)


class WOFOSTCropSimulationRunner(WOFOSTOptionsMixin):
    def __init__(self, model_name, workspace_dir="workspace"):
        """
        Initialize the runner.
        Calls update_workspace immediately to set up file paths.
        """
        self.model_name = model_name
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
    # 1. DATA PREPARATION (I/O Bound)
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

        # B. Soil Params (Check existence before processing)
        if force_update or not os.path.exists(self.files["soil_params"]):
            # Only ensure/download raw soil if we actually need to generate params
            soil_raw_path = self._ensure_soil(latitude, longitude, force_update)
            soil_df = pd.read_csv(soil_raw_path)
            self._save_params(WOFOSTSoilParameterProvider(soil_df), "soil_params")
        else:
            print("[PREP] Using existing Soil parameters.")
            
        # C. Site Params (Check existence before processing)
        if force_update or not os.path.exists(self.files["site_params"]):
            site_provider = WOFOSTSiteParametersProvider(self.model_name, **site_kwargs)
            # Ensure params are generated inside the provider if needed
            if hasattr(site_provider, 'get_params'): 
                site_provider.get_params()
            self._save_params(site_provider, "site_params")
        else:
            print("[PREP] Using existing Site parameters.")

        # D. Crop Params (Check existence before processing)
        if force_update or not os.path.exists(self.files["crop_params"]):
            self._save_params(
                WOFOSTCropParametersProvider(crop_name, variety_name), "crop_params"
            )
        else:
            print("[PREP] Using existing Crop parameters.")

        # E. Agromanagement Generation (Save to YAML)
        if force_update or not os.path.exists(self.files["agro"]):
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
        else:
            print("[PREP] Using existing Agromanagement config.")
            
        print("[PREP] System Ready.")

    # =========================================================================
    # 2. SIMULATION EXECUTION
    #    Pure logic. Reads prepared files and runs math.
    # =========================================================================
    def run_simulation(
        self,
        soil_overrides=None,
        site_overrides=None,
        crop_overrides=None,
        agro_file_path=None,
        output_vars=None
        
    ):
        """
        Runs the simulation.

        Args:
            soil_overrides (dict, optional): Override soil params.
            site_overrides (dict, optional): Override site params.
            crop_overrides (dict, optional): Override crop params.
            agro_file_path (str, optional): Path to a user-provided YAML file. If None, uses the workspace default.
            output_vars (list, optional): Variables to output.
        """
        print(f"[RUN] Initializing {self.model_name} in {self.workspace_dir}...")

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
        if not os.path.exists(self.files["weather"]) or not os.path.exists(agro_path):
            raise FileNotFoundError(
                "Missing weather or agro files. Run 'prepare_system' first."
            )
        
        weather_provider = ExcelWeatherDataProvider(self.files["weather"], force_reload=True)
        agromanagement = YAMLAgroManagementReader(agro_path)

        # 2. Load Parameters (Priority: DF Input -> CSV File -> Error)
        soil_dict = self._load_param_dict("soil_params", "value")
        site_dict = self._load_param_dict("site_params", "value")
        crop_dict = self._load_param_dict("crop_params", "default")
        
        if soil_overrides:
            soil_dict.update(soil_overrides)
        if site_overrides:
            site_dict.update(site_overrides)
        if crop_overrides:
            crop_dict.update(crop_overrides)
                        
        parameters = ParameterProvider(
            cropdata=crop_dict, soildata=soil_dict, sitedata=site_dict
        )
                
        # 3. Instantiate & Run
        try:
            ModelClass = get_model_class(self.model_name)
            if output_vars:
                wofsim = ModelClass(parameters, weather_provider, agromanagement, output_vars=output_vars)
            else:
                wofsim = ModelClass(parameters, weather_provider, agromanagement)
            wofsim.run_till_terminate()

            # 4. Save Output
            df_results = pd.DataFrame(wofsim.get_output())
            df_results.to_csv(self.files["output"], index=False)
            return df_results

        except Exception as e:
            raise RuntimeError(f"Simulation Failed: {e}")

    # =========================================================================
    # 3. INTERNAL HELPERS
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

    def _load_param_dict(self, file_key, value_col):
        """
        Helper to load parameters from disk.
        """
        # 1. Validate File Existence
        file_path = self.files[file_key]
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing parameter file: {file_path}")
        
        # 2. Load DataFrame from CSV
        df = pd.read_csv(file_path)
        
        # 3. Determine Column Name (fallback to 'default' if strict col missing)
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
            return self.files["weather"]

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
    
    # =========================================================================
    # 4. OPTIMIZATION
    # =========================================================================   
    def get_rerunner(self):
        """
        Returns a WOFOSTFastEngine instance instead of a closure.
        """
        # Validation & Pre-loading
        if not os.path.exists(self.files['weather']):
            raise FileNotFoundError("Run prepare_system first.")
        
        # Load the path
        weather = self.files['weather']
        agro = self.files['agro']
        
        # Load Dictionaries
        base_params = {
            'soil': self._load_param_dict("soil_params", "value"),
            'site': self._load_param_dict("site_params", "value"),
            'crop': self._load_param_dict("crop_params", "default")
        }

        # Return the class instance
        return _WOFOSTLazyEngine(self.model_name, weather, agro, base_params)


def _WOFOST_prepare_batch_system(kwargs):
    """
    Worker 1: PREPARATION (Downloads & File Generation).
    """
    try:
        point_id = kwargs['id']
        model_name = kwargs['model_name']
        unique_dir = os.path.join(kwargs['base_workspace_dir'], f"point_{int(point_id)}")
        
        # Instantiate Runner locally
        runner = WOFOSTCropSimulationRunner(model_name, workspace_dir=unique_dir)
        
        # Execute Preparation Logic
        runner.prepare_system(
            latitude=kwargs['latitude'],
            longitude=kwargs['longitude'],
            campaign_start=kwargs['campaign_start'],
            campaign_end=kwargs['campaign_end'],
            crop_start=kwargs['crop_start'],
            crop_end=kwargs['crop_end'],
            crop_name=kwargs['crop_name'],
            variety_name=kwargs['variety_name'],
            crop_start_type=kwargs.get('crop_start_type', 'emergence'),
            crop_end_type=kwargs.get('crop_end_type', 'harvest'),
            max_duration=kwargs.get('max_duration', 300),
            timed_events=kwargs.get('timed_events', None),
            state_events=kwargs.get('state_events', None),
            force_update=kwargs.get('force_update', False),
            **kwargs.get('site_kwargs', {})
        )
        return {"id": point_id, "status": "Success"}
        
    except Exception as e:
        return {"id": kwargs.get('id'), "status": "Failed", "error": str(e)}
    
    
def _WOFOST_run_batch_task(kwargs):
    """
    Worker 2: EXECUTION (Simulation Only).
    """
    try:
        point_id = kwargs['id']
        model_name = kwargs['model_name']
        unique_dir = os.path.join(kwargs['base_workspace_dir'], f"point_{int(point_id)}")
        
        # Instantiate Runner locally
        runner = WOFOSTCropSimulationRunner(model_name, workspace_dir=unique_dir)
        
        # Execute Simulation Logic
        df_result = runner.run_simulation(
            soil_overrides=kwargs.get('soil_overrides'), # Optional overrides
            site_overrides=kwargs.get('site_overrides'),
            crop_overrides=kwargs.get('crop_overrides'),
            agro_file_path=kwargs.get('agro_file_path'),
            output_vars=kwargs.get('output_vars')
        )
        
        # Tag Results
        df_result['point_id'] = point_id
        df_result['latitude'] = kwargs['latitude']
        df_result['longitude'] = kwargs['longitude']
        
        return df_result.reset_index()

    except Exception as e:
        print(f"Error running point {kwargs.get('id')}: {e}")
        return None


class WOFOSTCropSimulationBatchRunner(WOFOSTOptionsMixin):
    def __init__(self, model_name, locations_csv_path, workspace_dir="batch_workspace"):
        self.model_name = model_name
        self.workspace_dir = os.path.abspath(workspace_dir)
        os.makedirs(self.workspace_dir, exist_ok=True)

        self.locations_csv_path = locations_csv_path
        self.locations_df = pd.read_csv(locations_csv_path)
        self.locations_df['id'] = self.locations_df['id'].astype(int)
        
    # =========================================================================
    # PHASE 1: PARALLEL SYSTEM PREPARATION
    # =========================================================================
    def prepare_batch_system(self, 
                             max_workers=4,
                             **preparation_args):
        """
        Runs `prepare_system` for all points in parallel.
        Downloads data and creates config files for every point.
        """
        tasks = []
        print(f"[BATCH PREP] Preparing tasks for {len(self.locations_df)} locations...")
        
        # 1. Build Task List
        for _, row in self.locations_df.iterrows():
            # Separate site_kwargs from the main args
            site_kwargs = {k: v for k, v in preparation_args.items() if k not in [
                'campaign_start', 'campaign_end', 'crop_start', 'crop_end',
                'crop_name', 'variety_name', 'force_update',
                'crop_start_type', 'crop_end_type', 'max_duration', 
                'timed_events', 'state_events'
            ]}
            
            task_payload = {
                'id': int(row['id']),
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'base_workspace_dir': self.workspace_dir,
                'model_name': self.model_name,
                'site_kwargs': site_kwargs,
                **preparation_args # Pass rest of args (dates, crop, etc)
            }
            tasks.append(task_payload)

        # 2. Execute Parallel Preparation
        print(f"[BATCH PREP] Starting preparation with {max_workers} workers...")
        success_count = 0
        
        with multiprocessing.Pool(processes=max_workers, maxtasksperchild=1) as pool:
            iterator = pool.imap_unordered(_WOFOST_prepare_batch_system, tasks)
            
            for result in tqdm(iterator, total=len(tasks), desc="Preparing Data"):
                if result['status'] == "Success":
                    success_count += 1
                else:
                    print(f"FAILED Point {result['id']}: {result.get('error')}")

        print(f"[BATCH PREP] Completed. {success_count}/{len(tasks)} locations ready.")

    # =========================================================================
    # PHASE 2: PARALLEL EXECUTION
    # =========================================================================
    def run_batch_simulation(self, 
                             max_workers=4,
                             soil_overrides=None, 
                             site_overrides=None, 
                             crop_overrides=None,
                             agro_file_path=None,
                             output_vars=None):
        """
        Runs `run_simulation` for all points in parallel.
        Assumes data is already prepared.
        
        Args:
            max_workers (int, optional): Number of CPU cores to use.
            soil_overrides (dict, optional): Override soil params.
            site_overrides (dict, optional): Override site params.
            crop_overrides (dict, optional): Override crop params.
            agro_file_path (str, optional): Path to a user-provided YAML file. If None, uses the workspace default.
            output_vars (list, optional): Variables to output.
        """
        tasks = []
        
        print(f"[BATCH RUN] Preparing execution tasks for {len(self.locations_df)} locations...")
        
        # 1. Build Task List
        for _, row in self.locations_df.iterrows():
            task_payload = {
                'id': row['id'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'base_workspace_dir': self.workspace_dir,
                'model_name': self.model_name,
                # Optional overrides passed to all workers
                'soil_overrides': soil_overrides,
                'site_overrides': site_overrides,
                'crop_overrides': crop_overrides,
                'agro_file_path': agro_file_path,
                'output_vars': output_vars
            }
            tasks.append(task_payload)

        # 2. Execute Parallel Simulation
        results_list = []
        print(f"[BATCH RUN] Starting simulation with {max_workers} workers...")

        with multiprocessing.Pool(processes=max_workers, maxtasksperchild=1) as pool:
            iterator = pool.imap(_WOFOST_run_batch_task, tasks)
            
            for res_df in tqdm(iterator, total=len(tasks), desc="Simulating"):
                if res_df is not None:
                    results_list.append(res_df)
                    
        # 3. Combine & Save
        if results_list:
            final_df = pd.concat(results_list, ignore_index=True)
            if 'index' in final_df.columns:
                final_df.drop('index', axis=1, inplace=True)
            
            final_df['point_id'] = final_df['point_id'].astype(int)
            
            output_path = os.path.join(self.workspace_dir, "batch_results.csv")
            final_df.to_csv(output_path, index=False)
            print(f"[BATCH RUN] Success! Results saved to {output_path}")
            return final_df
        else:
            print("[BATCH RUN] Failed. No results generated.")
            return pd.DataFrame()
        
    # =========================================================================
    # OPTIMIZATION
    # =========================================================================
    def get_batch_rerunners(self):
        """
        Initializes a fast re-runner for EVERY location in the batch.
        Loads all weather/soil data into RAM once.
        
        Returns:
            dict: { location_id: fast_runner_function }
        """
        batch_runners = {}
        print(f"[BATCH INIT] Pre-loading data for {len(self.locations_df)} locations...")
        
        for _, row in self.locations_df.iterrows():
            loc_id = int(row['id'])
            point_dir = os.path.join(self.workspace_dir, f"point_{loc_id}")
            
            # Initialize the single runner for this specific point
            runner = WOFOSTCropSimulationRunner(model_name=self.model_name, workspace_dir=point_dir)
            
            try:
                engine_object = runner.get_rerunner()
                batch_runners[loc_id] = engine_object
            except Exception as e:
                print(f"[WARN] Skipping point {loc_id}: {e}")
                
        return batch_runners
    
    
class _WOFOSTLazyEngine:
    """
    A picklable worker class.
    """
    def __init__(self, model_name, weather_path, agro_path, base_params):
        self.model_name = model_name
        self.weather_path = weather_path 
        self.agro_path = agro_path    
        self.base_params = base_params    

    def __call__(self, crop_overrides=None, soil_overrides=None, site_overrides=None):
        """
        This method makes the object behave like a function: engine(...)
        """
        try:
            # Weather data loading
            weather = ExcelWeatherDataProvider(self.weather_path, force_reload=True)
            # Agromanagement Loading
            agro = YAMLAgroManagementReader(self.agro_path)
            
        except Exception as e:
            # If IO fails, return None so the worker doesn't crash
            return None
        
        # 1. Merge Parameters (Fast In-Memory)
        current_crop = self.base_params['crop'].copy()
        if crop_overrides: current_crop.update(crop_overrides)
        
        current_soil = self.base_params['soil'].copy()
        if soil_overrides: current_soil.update(soil_overrides)
        
        current_site = self.base_params['site'].copy()
        if site_overrides: current_site.update(site_overrides)
        
        parameters = ParameterProvider(
            cropdata=current_crop, soildata=current_soil, sitedata=current_site
        )

        try:
            ModelClass = get_model_class(self.model_name)
            wofsim = ModelClass(parameters, weather, agro)
            wofsim.run_till_terminate()
            return pd.DataFrame(wofsim.get_output())
        except Exception:
            return None