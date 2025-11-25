"""Module to prepare weather data"""

import numpy as np
import pandas as pd
import ee
import yaml
import importlib.resources as pkg_resources
from . import configs
from geeagri.extract import extract_timeseries_to_point, extract_timeseries_to_polygon
from .conversions import CONVERSION_FUNCS


class ClimateConfig:
    """Parses and holds configuration for climate variables."""

    def __init__(self, config_dict):
        self.raw = config_dict
        self.variables = config_dict["variables"]
        self.all_bands = []
        self.var_to_bands = {}
        self.var_to_units = {}
        self.var_to_conversion = {}
        self.derived = set()
        self._parse_variables()

    def _parse_variables(self):
        for var_name, info in self.variables.items():
            self.var_to_units[var_name] = (
                info.get("native_unit"),
                info.get("target_unit"),
            )
            self.var_to_conversion[var_name] = info.get("conversion")
            if info.get("derived", False):
                self.derived.add(var_name)

            bands = []
            for key, value in info.items():
                if key.startswith("band") and value is not None:
                    bands.append(value)
                    self.all_bands.append(value)
            self.var_to_bands[var_name] = bands

    def get_all_bands(self):
        return list(set(self.all_bands))

    def is_derived(self, var_name):
        return var_name in self.derived


class GEEWeatherDataProvider:
    """
    Handles data retrieval, processing, and export of weather data from Google Earth Engine in PCSE format.

    IMPORTANT: This class strictly handles POINT data. If a Geometry/Polygon is provided,
    it extracts data for the CENTROID of that geometry only.

    Args:
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
        latitude (float, optional): Latitude (if geometry not provided).
        longitude (float, optional): Longitude (if geometry not provided).
        geometry (ee.Geometry, optional): Polygon or geometry object. Will be converted to its Centroid.
        source (str): Key in meteo.yaml (e.g., 'era5_land').
        filepath (str, optional): Default output path.
        ee_project (str, optional): GCloud project ID for GEE initialization.
        **site_kwargs: Extra metadata for the Excel header (e.g., Station, Country).
    """

    def __init__(
        self,
        start_date,
        end_date,
        latitude=None,
        longitude=None,
        geometry=None,
        source="era5_land",
        filepath=None,
        ee_project=None,
        **site_kwargs,
    ):

        self._check_gee_initialized(ee_project)

        if geometry:
            if isinstance(geometry, (ee.Feature, ee.FeatureCollection)):
                region_geom = geometry.geometry()
            else:
                region_geom = ee.Geometry(geometry)

            # Calculate Centroid
            try:
                centroid_obj = region_geom.centroid(maxError=1)
                coords = centroid_obj.coordinates().getInfo()
            except Exception:
                centroid_obj = region_geom.bounds(maxError=1).centroid(maxError=1)
                coords = centroid_obj.coordinates().getInfo()

            self.longitude = coords[0]
            self.latitude = coords[1]

        elif latitude is not None and longitude is not None:
            self.latitude = latitude
            self.longitude = longitude
        else:
            raise ValueError(
                "Must provide either 'geometry' OR 'latitude' and 'longitude'."
            )

        self.region = ee.Geometry.Point([self.longitude, self.latitude])

        self.start_date = start_date
        self.end_date = end_date
        self.source = source.lower()
        self.filepath = filepath
        self.site_kwargs = site_kwargs

        self._cached_df = None
        self._cached_elevation = None

        with pkg_resources.files(configs).joinpath("meteo.yaml").open("r") as f:
            full_config = yaml.safe_load(f)

        if source.lower() not in full_config:
            raise ValueError(
                f"Source '{source}' not found. Available: {list(full_config.keys())}"
            )

        self.weather_config = full_config[source.lower()]
        self.cfg = ClimateConfig(self.weather_config)

    def _check_gee_initialized(self, project=None):
        """
        Checks if GEE is initialized. If not, attempts to initialize.
        """
        try:
            ee.Image(0).getInfo()
        except Exception:
            print("GEE not initialized. Attempting initialization...")
            try:
                # Try initializing with specific project if provided, else default
                if project:
                    ee.Initialize(project=project)
                else:
                    ee.Initialize()
                print("GEE Initialized successfully.")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize Earth Engine: {e}.\n"
                    "Please run 'earthengine authenticate' in your terminal first."
                )

    def _get_elevation(self):

        if self._cached_elevation is not None:
            return self._cached_elevation

        try:
            geom = self.region
            dem_source = self.weather_config.get(
                "dem_source", "projects/sat-io/open-datasets/GLO-30"
            )

            elev = (
                ee.ImageCollection(dem_source)
                .filterBounds(geom)
                .first()
                .sample(geom, scale=30)
                .first()
                .get("b1")
            )

            val = elev.getInfo()
            self._cached_elevation = round(float(val), 3) if val is not None else 0.0
            return self._cached_elevation

        except Exception as e:
            print(f"Warning: Could not fetch elevation ({e}). Defaulting to 0.")
            self._cached_elevation = 0.0
            return 0.0

    def _extract_data(self):
        """
        Extraction of weather data.
        """
        band_names = self.cfg.get_all_bands()
        scale = self.weather_config.get("default_scale", 5000)
        collection_id = self.weather_config.get("collection")

        ic = ee.ImageCollection(collection_id)

        return extract_timeseries_to_point(
            lat=self.latitude,
            lon=self.longitude,
            image_collection=ic,
            start_date=self.start_date,
            end_date=self.end_date,
            band_names=band_names,
            scale=scale,
        )

    def get_data(self):
        if self._cached_df is not None:
            return self._cached_df

        df_raw = self._extract_data()
        output = pd.DataFrame(index=df_raw.index)
        output["date"] = df_raw["time"]

        for var, bands in self.cfg.var_to_bands.items():
            conversion = self.cfg.var_to_conversion.get(var)
            converter_func = CONVERSION_FUNCS.get(conversion, lambda x: x)

            inputs = [df_raw[b] for b in bands]

            try:
                output[var] = converter_func(*inputs)
            except TypeError as e:
                raise ValueError(
                    f"Error calculating {var}: Function '{conversion}' "
                    f"expected different arguments than provided bands {bands}. "
                    f"Details: {e}"
                )

        self._cached_df = output.round(3)
        return self._cached_df

    def save_weather_excel(self, filepath=None, **override_kwargs):
        target_path = filepath or self.filepath
        if not target_path:
            raise ValueError("Invalid filepath.")

        df = self.get_data()

        meta_defaults = {
            "Country": "Unknown",
            "Station": "Unknown",
            "Description": self.weather_config.get("description"),
            "Source": self.weather_config.get("collection"),
            "Contact": "Unknown",
            "Missing values": -999,
            "AngstromA": 0.25,
            "AngstromB": 0.50,
            "HasSunshine": False,
        }

        meta = {**meta_defaults, **self.site_kwargs, **override_kwargs}

        excel_rows = []

        excel_rows.append(["Site Characteristics"])
        excel_rows.append(["Country", meta["Country"]])
        excel_rows.append(["Station", meta["Station"]])
        excel_rows.append(["Description", meta["Description"]])
        excel_rows.append(["Source", meta["Source"]])
        excel_rows.append(["Contact", meta["Contact"]])
        excel_rows.append(["Missing values", meta["Missing values"]])

        excel_rows.append(
            [
                "Longitude",
                "Latitude",
                "Elevation",
                "AngstromA",
                "AngstromB",
                "HasSunshine",
            ]
        )
        excel_rows.append(
            [
                self.longitude,
                self.latitude,
                self._get_elevation(),
                meta["AngstromA"],
                meta["AngstromB"],
                str(meta["HasSunshine"]).upper(),
            ]
        )

        excel_rows.append(["Observed data"])

        var_order = ["IRRAD", "TMIN", "TMAX", "VAP", "WIND", "RAIN", "SNOWDEPTH"]
        present_vars = [v for v in var_order if v in df.columns]

        header_names = ["DAY"] + present_vars
        excel_rows.append(header_names)

        header_units = ["date"]
        for v in present_vars:
            if v in self.cfg.var_to_units:
                header_units.append(self.cfg.var_to_units[v][1])
            else:
                header_units.append("-")
        excel_rows.append(header_units)

        df_export = df.copy()
        df_export = df_export.fillna(meta["Missing values"])
        df_export = df_export[["date"] + present_vars]

        with pd.ExcelWriter(target_path, engine="openpyxl") as writer:
            pd.DataFrame(excel_rows).to_excel(
                writer, sheet_name="Sheet1", index=False, header=False, startrow=0
            )

            df_export.to_excel(
                writer,
                sheet_name="Sheet1",
                index=False,
                header=False,
                startrow=len(excel_rows),
            )

        print(f"File saved successfully to {target_path}")
