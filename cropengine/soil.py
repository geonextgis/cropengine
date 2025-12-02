"""Module to prepare soil data"""

import requests
import numpy as np
import pandas as pd
import yaml
import pedon as pe
from scipy.optimize import curve_fit
import os
import importlib.resources as pkg_resources
from . import configs

class IsricSoilDataProvider:
    """
    Initialize the ISRIC Soil Data Provider.

    Args:
        latitude (float): The latitude of the location.
        longitude (float): The longitude of the location.
        properties (list, optional): List of soil properties to fetch (e.g., ['clay', 'sand']). 
                                         Defaults to all available in config.
        depths (list, optional): List of depth ranges (e.g., ['0-5cm']). 
                                     Defaults to all available in config.
        values (list, optional): List of statistical values (e.g., ['mean', 'Q0.5']). 
                                     Defaults to all available in config.
        filepath (str, optional): Path to save the resulting CSV file. Defaults to None.
    """

    def __init__(self, latitude, longitude, properties=None, depths=None, values=None, filepath=None):
        self.latitude = latitude
        self.longitude = longitude
        self.filepath = filepath
        
        # Internal cache storage
        self._cached_df = None
        
        # Load configuration using pkg_resources
        with pkg_resources.files(configs).joinpath("soil.yaml").open("r") as f:
            full_config = yaml.safe_load(f)
            
        self.config = full_config['isric_rest_api']
        self.base_url = self.config['api']['base_url']
        
        # Validate inputs
        self.properties = self._validate_input(properties, self.config['options']['properties'], "Property")
        self.depths = self._validate_input(depths, self.config['options']['depths'], "Depth")
        self.values = self._validate_input(values, self.config['options']['values'], "Value")

        self.query = {
            "lat": latitude,
            "lon": longitude,
            "property": self.properties,
            "depth": self.depths,
            "value": self.values
        }
        
    def _validate_input(self, user_input, valid_options, category_name):
        """Validates user input against the loaded config options."""
        if user_input is None:
            return valid_options

        if isinstance(user_input, str):
            user_input = [user_input]

        # Check for invalid items
        invalid_items = [item for item in user_input if item not in valid_options]

        if invalid_items:
            error_msg = (
                f"\nError: Invalid {category_name}(s) provided: {invalid_items}\n"
                f"Available options in {category_name} are: {valid_options}"
            )
            raise ValueError(error_msg)
            
        return user_input
    
    def _extract_data(self):
        """Internal method to hit the API."""
        print(f"Fetching soil data for {self.latitude}, {self.longitude}...")
        try:
            response = requests.get(self.base_url, params=self.query)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
        
    def get_data(self):
        """
        Fetches, parses, and returns soil data. 
        Uses cached memory if data has already been fetched for this instance.
        """
        # 1. Check Cache (Optimization)
        if self._cached_df is not None:
            print("Returning cached data (no API call made).")
            return self._cached_df

        # 2. Fetch Data
        raw_data = self._extract_data()
        
        if not raw_data:
            return pd.DataFrame()

        rows = []
        layers = raw_data.get('properties', {}).get('layers', [])

        # 3. Parse Data
        for layer in layers:
            prop_name = layer.get('name')
            
            # Get Unit Transformation details
            unit_info = layer.get('unit_measure', {})
            d_factor = unit_info.get('d_factor', 1) 
            
            # Grab both unit labels
            mapped_unit = unit_info.get('mapped_units', 'unknown') # e.g. "cg/cm³"
            target_unit = unit_info.get('target_units', 'unknown') # e.g. "kg/dm³"

            for depth_record in layer.get('depths', []):
                depth_range = depth_record.get('label')
                values_dict = depth_record.get('values', {})
                
                for metric, raw_val in values_dict.items():
                    # Calculate transformed value
                    if isinstance(raw_val, (int, float)) and d_factor != 0:
                        converted_val = raw_val / d_factor
                    else:
                        converted_val = raw_val

                    rows.append({
                        'latitude': self.latitude,
                        'longitude': self.longitude,
                        'property': prop_name,
                        'depth': depth_range,
                        'metric': metric,
                        # Raw Data (Integer from API)
                        'value': raw_val,
                        'unit': mapped_unit,
                        # Transformed Data (Scientific Float)
                        'transformed_value': converted_val,
                        'transformed_unit': target_unit
                    })

        df = pd.DataFrame(rows)
        
        # 4. Update Cache
        self._cached_df = df

        # 5. Save to File (if requested)
        if self.filepath and not df.empty:
            try:
                df.to_csv(self.filepath, index=False)
                print(f"Data saved to {self.filepath}")
            except Exception as e:
                print(f"Failed to save file: {e}")

        return df

class WOFOSTSoilParameterProvider(dict):
    """
    Calculates soil physics and chemical parameters required for WOFOST 
    crop modeling using ISRIC SoilGrids data.

    This class extracts soil properties, estimates hydraulic conductivity 
    using Pedotransfer Functions (Cosby), and fits the Van Genuchten 
    equation.
    
    Args:
        soil_data (pd.DataFrame): DataFrame containing ISRIC SoilGrids data.
        **kwargs: Optional overrides for specific soil parameters.
    """
    
    # Default parameters for potential production
    _defaults = {
        "SMFCF": 0.3,
        "SM0": 0.4,
        "SMW": 0.1,
        "RDMSOL": 120,
        "CRAIRC": 0.06,
        "K0": 10.,
        "SOPE": 10.,
        "KSUB": 10.
    }

    def __init__(self, soil_data, **kwargs):
        dict.__init__(self)
        
        self.df = soil_data
        self.params = {}
        
        # 1. Load Valid Keys dynamically from YAML
        self.param_metadata = get_wofost_soil_parameters_metadata()
        self.valid_keys = set(self.param_metadata.keys())
        
        # 2. Validate kwargs immediately
        unknown_keys = [k for k in kwargs.keys() if k not in self.valid_keys]
        if unknown_keys:
            raise ValueError(
                f"Invalid WOFOST parameters provided: {unknown_keys}. "
                "These keys are not defined in the soil configuration."
            )
            
        self.overrides = kwargs
        
        # Specific pF points required by WOFOST/PCSE implementations
        self.pf_range = np.array([-1.000, 1.000, 1.300, 1.491, 2.000, 2.400, 2.700, 3.400, 4.204, 6.000])
        self.h_range = 10**self.pf_range

    def _get_val(self, prop_name):
        """Helper to safely extract a specific property value from the DataFrame."""
        try:
            return self.df.loc[self.df['property'] == prop_name, 'value'].values[0]
        except IndexError:
            raise ValueError(f"Missing required property: {prop_name}")

    def _calculate_ksat_ptf(self, sand_pct, clay_pct):
        """Estimate Ksat (cm/day) using Cosby (1984) PTF."""
        log_ksat = -0.6 + (0.0126 * sand_pct) - (0.0064 * clay_pct)
        return max(0.1, min((10**log_ksat) * 24, 500.0))

    def _van_genuchten_theta(self, h, theta_r, alpha, n, theta_s):
        """Mualem-Van Genuchten equation."""
        h = np.maximum(h, 0.0) 
        m = 1 - (1 / n)
        return theta_r + (theta_s - theta_r) / ((1 + (alpha * h)**n)**m)

    def get_params(self):
        """
        Calculates parameters, fits curves, applies overrides, and returns dictionary.
        """
        # 1. Extract and Convert Data
        bdod = self._get_val('bdod') / 100.0 
        soc_g_kg = self._get_val('soc') / 10.0
        n_g_kg = self._get_val('nitrogen') / 100.0
        ph = self._get_val('phh2o') / 10.0
        sand_pct = self._get_val('sand') / 10.0
        clay_pct = self._get_val('clay') / 10.0
        
        theta_10 = self._get_val('wv0010') / 1000.0 
        theta_33 = self._get_val('wv0033') / 1000.0
        theta_1500 = self._get_val('wv1500') / 1000.0
        
        porosity = 1 - (bdod / 2.65)
        ksat_est = self._calculate_ksat_ptf(sand_pct, clay_pct)

        # 2. Fit Retention Curve
        h_obs = np.array([0.01, 100.0, 330.0, 15000.0]) 
        theta_obs = np.array([porosity, theta_10, theta_33, theta_1500])
        p0 = [0.01, 0.01, 1.5]
        bounds = ([0.0, 1e-5, 1.01], [theta_1500, 10.0, 10.0])

        try:
            popt, _ = curve_fit(
                lambda h, tr, a, n: self._van_genuchten_theta(h, tr, a, n, theta_s=porosity),
                h_obs, theta_obs, p0=p0, bounds=bounds, method='trf'
            )
            theta_r_opt, alpha_opt, n_opt = popt
        except Exception as e:
            print(f"Scipy fitting failed: {e}. Using default texture-based parameters.")
            theta_r_opt, alpha_opt, n_opt = 0.01, 0.01, 1.5

        # 3. Create Pedon Model
        vg_model = pe.soilmodel.Genuchten(
            k_s=ksat_est, theta_s=porosity, theta_r=theta_r_opt,
            alpha=alpha_opt, n=n_opt, l=0.5
        )

        # 4. Generate Tables & Scalars
        theta_curve = vg_model.theta(self.h_range)
        k_curve = vg_model.k(self.h_range)
        
        sm_table = []
        cond_table = []
        for pf, th, k in zip(self.pf_range, theta_curve, k_curve):
            sm_table.extend([float(pf), float(th)])
            k_safe = max(k, 1e-15) 
            cond_table.extend([float(pf), float(np.log10(k_safe))])

        sm0 = vg_model.theta(0.01)
        smfcf = vg_model.theta(100.0)
        smw = vg_model.theta(16000.0)
        crairc = max(0.05, sm0 - smfcf)

        # 5. Chemical Props & Dimensions
        fsomi = (soc_g_kg * 1.724) / 1000.0
        cn_ratio = soc_g_kg / n_g_kg if n_g_kg > 0 else 10.0
        
        depth_str = self.df.iloc[0]['depth']
        try:
            top, bottom = depth_str.replace('cm', '').split('-')
            thickness = float(bottom) - float(top)
        except:
            thickness = 10.0

        # 6. Construct Final Dictionary
        self.params = self._defaults.copy()
        
        # Overwrite with calculated values
        self.params.update({
            'Soil_pH': float(round(ph, 3)),
            'RHOD': float(round(bdod, 3)),
            'FSOMI': float(round(fsomi, 3)),
            'CNRatioSOMI': float(round(cn_ratio, 3)),
            'Thickness': float(round(thickness, 3)),
            'SM0': float(round(float(sm0), 3)),
            'SMFCF': float(round(float(smfcf), 3)),
            'SMW': float(round(float(smw), 3)),
            'CRAIRC': float(round(float(crairc), 3)),
            'SMfromPF': [round(x, 3) for x in sm_table],
            'CONDfromPF': [round(x, 3) for x in cond_table]
        })
        
        # 7. Apply Validated Overrides
        if self.overrides:
            self.params.update(self.overrides)
        
        return self.params

def get_wofost_soil_parameters_metadata():
    """
    Parses 'configs/soil_params.yaml' to extract soil parameter definitions.

    Returns:
        dict: Dictionary of parameters with 'description' and 'unit'.
    """
    metadata = {}

    try:
        with pkg_resources.files(configs).joinpath("soil_params.yaml").open("r") as f:
            full_config = yaml.safe_load(f)
    except (FileNotFoundError, ModuleNotFoundError) as e:
        raise ValueError(f"Could not load 'soil_params.yaml' from 'configs'. Error: {e}")
    except yaml.YAMLError as exc:
        raise ValueError(f"Error parsing YAML content: {exc}")

    # Locate the root soil parameters section
    soil_section = full_config.get('wofost', {}).get('soil_params', {})
    
    if not soil_section:
        if 'WaterbalanceFD' in full_config or 'WaterBalanceLayered' in full_config:
             soil_section = full_config
        else:
            return {}

    def _recursive_extract(node):
        for key, value in node.items():
            if isinstance(value, dict):
                if 'description' in value or 'unit' in value:
                    metadata[key] = {
                        'description': value.get('description', 'No description available'),
                        'unit': value.get('unit', '-')
                    }
                else:
                    _recursive_extract(value)

    _recursive_extract(soil_section)
    
    # Ensure defaults exist
    if 'K0' not in metadata:
        metadata['K0'] = {'description': 'Hydraulic conductivity of saturated soil', 'unit': 'cm/day'}
    if 'IVINF' not in metadata:
        metadata['IVINF'] = {'description': 'Infiltration limiter', 'unit': '-'}

    return metadata