"""Module to setup agromanagement"""

import yaml
import datetime
import importlib.resources as pkg_resources
from . import configs
from typing import List, Dict, Optional, Union


class WOFOSTAgroEventBuilder:
    """
    Helper class to build PCSE agromanagement events using a YAML schema for validation.
    """

    def __init__(self):
        try:
            with pkg_resources.files(configs).joinpath("agromanagement.yaml").open(
                "r"
            ) as f:
                self.schema = yaml.safe_load(f)["wofost"]
        except Exception as e:
            raise RuntimeError(f"Failed to load agromanagement.yaml: {e}")

    def get_timed_events_info(self):
        return self.schema["TimedEvents"]

    def get_state_events_info(self):
        return self.schema["StateEvents"]

    def _convert_date(
        self, date_val: Union[str, datetime.date, None]
    ) -> Optional[datetime.date]:
        """Helper to convert string dates to datetime.date objects."""
        if date_val is None:
            return None
        if isinstance(date_val, str):
            return datetime.datetime.strptime(date_val, "%Y-%m-%d").date()
        if isinstance(date_val, (datetime.date, datetime.datetime)):
            return date_val
        raise ValueError(
            f"Invalid date format: {date_val}. Expected YYYY-MM-DD string or datetime.date object."
        )

    def create_timed_events(self, signal_type: str, events_list: List[Dict]) -> dict:
        """
        Creates a single TimedEvent dictionary containing a LIST of dates.
        """
        if signal_type not in self.schema["TimedEvents"]:
            raise ValueError(f"Unknown TimedEvent signal: {signal_type}")

        schema_def = self.schema["TimedEvents"][signal_type]
        required_params = schema_def["events_table"].keys()

        populated_events_list = []

        for entry in events_list:
            current_date = self._convert_date(entry["event_date"])

            params = {}
            for param in required_params:
                params[param] = entry[param]

            populated_events_list.append({current_date: params})

        return {
            "event_signal": signal_type,
            "name": schema_def.get("name"),
            "comment": schema_def.get("comment"),
            "events_table": populated_events_list,
        }

    def create_state_events(
        self,
        signal_type: str,
        state_var: str,
        zero_condition: str,
        events_list: List[Dict],
    ) -> dict:
        """
        Creates a single StateEvent dictionary containing a LIST of thresholds.
        """
        if signal_type not in self.schema["StateEvents"]:
            raise ValueError(f"Unknown StateEvent signal: {signal_type}")

        schema_def = self.schema["StateEvents"][signal_type]
        required_params = schema_def["events_table"].keys()

        # Change: Use a LIST for the events table
        populated_events_list = []

        for entry in events_list:
            threshold = entry["threshold"]

            params = {}
            for param in required_params:
                params[param] = entry[param]

            # Append as a single-key dictionary to the list
            populated_events_list.append({threshold: params})

        return {
            "event_signal": signal_type,
            "event_state": state_var,
            "zero_condition": zero_condition,
            "name": schema_def.get("name"),
            "comment": schema_def.get("comment"),
            "events_table": populated_events_list,
        }


class WOFOSTAgroManagementProvider(list):
    """
    A dynamic provider for WOFOST AgroManagement.
    Generates a rotation of crops based on start/end dates and handles YAML serialization.
    """

    def __init__(self):
        super().__init__()

    def _convert_date(
        self, date_val: Union[str, datetime.date, None]
    ) -> Optional[datetime.date]:
        """Helper to convert string dates to datetime.date objects."""
        if date_val is None:
            return None
        if isinstance(date_val, str):
            return datetime.datetime.strptime(date_val, "%Y-%m-%d").date()
        if isinstance(date_val, (datetime.date, datetime.datetime)):
            return date_val
        raise ValueError(
            f"Invalid date format: {date_val}. Expected YYYY-MM-DD string or datetime.date object."
        )

    def add_campaign(
        self,
        campaign_start_date: Union[str, datetime.date],
        campaign_end_date: Union[str, datetime.date],
        crop_name: str,
        variety_name: str,
        crop_start_date: Union[str, datetime.date],
        crop_end_date: Optional[Union[str, datetime.date]] = None,
        crop_start_type: str = "sowing",
        crop_end_type: str = "maturity",
        max_duration: int = 300,
        timed_events: List[Dict] = None,
        state_events: List[Dict] = None,
    ):
        """
        Adds a single cropping campaign to the rotation.

        Args:
            campaign_start_date: Start date of the campaign (str 'YYYY-MM-DD' or date object).
            campaign_end_date: End date of the campaign (str 'YYYY-MM-DD' or date object).
            crop_name: Name of the crop (e.g., 'wheat').
            variety_name: Variety identifier (e.g., 'winter-wheat').
            crop_start_date: Date of sowing or emergence (str 'YYYY-MM-DD' or date object).
            crop_end_date: Optional harvest date (str 'YYYY-MM-DD' or date object).
            crop_start_type: 'sowing' or 'emergence'.
            crop_end_type: 'maturity', 'harvest', or 'earliest'.
            max_duration: Maximum duration of the crop cycle in days.
            timed_events: List of timed event dictionaries (from EventBuilder).
            state_events: List of state event dictionaries (from EventBuilder).
        """
        # Convert inputs to ensure they are date objects or valid strings
        c_start = self._convert_date(campaign_start_date)
        c_end = self._convert_date(campaign_end_date)
        crop_start = self._convert_date(crop_start_date)
        crop_end = self._convert_date(crop_end_date) if crop_end_date else None

        self._last_campaign_end = c_end

        # 1. Define the base CropCalendar
        crop_calendar = {
            "crop_name": crop_name,
            "variety_name": variety_name,
            "crop_start_date": crop_start,
            "crop_start_type": crop_start_type,
            "crop_end_type": crop_end_type,
            "max_duration": max_duration,
        }

        # 2. Conditionally add crop_end_date only if it exists
        if crop_end is not None:
            crop_calendar["crop_end_date"] = crop_end

        # 3. Build the full config
        campaign_config = {
            "CropCalendar": crop_calendar,
            "TimedEvents": timed_events if timed_events else None,
            "StateEvents": state_events if state_events else None,
        }

        # Append the campaign dictionary {start_date: config} to the list
        self.append({c_start: campaign_config})

    def add_trailing_empty_campaign(self):
        """
        Adds a final empty campaign to ensure the simulation runs until the very end
        of the requested period.

        Args:
            start_date: Start date of the empty period (str 'YYYY-MM-DD' or date object).
        """
        if self._last_campaign_end is None:
            raise RuntimeError(
                "Cannot add trailing empty campaign before adding at least one campaign."
            )

        self.append({self._last_campaign_end: None})

    def save_to_yaml(self, filename: str):
        """
        Exports the current agromanagement configuration to a YAML file.

        Structure matches the PCSE requirement:
        AgroManagement:
        - Date:
            CropCalendar: ...
            TimedEvents: ...
        """
        # Wrap the list in the root 'AgroManagement' key
        output_structure = {"AgroManagement": list(self)}

        with open(filename, "w") as f:
            yaml.dump(output_structure, f, sort_keys=False)
