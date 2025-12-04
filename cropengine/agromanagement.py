"""Module to setup agromanagement"""

import datetime
from typing import List, Dict, Optional

class WOFOSTAgroManagementProvider(list):
    """
    A dynamic provider for WOFOST AgroManagement.
    Generates a rotation of crops based on start/end dates.
    """
    def __init__(self):
        super().__init__()

    def add_campaign(
        self,
        campaign_start_date: datetime.date,
        crop_name: str,
        variety_name: str,
        crop_start_date: datetime.date,
        crop_end_date: Optional[datetime.date] = None,
        crop_start_type: str = "sowing",
        crop_end_type: str = "maturity",
        max_duration: int = 300,
        timed_events: List[Dict] = None,
        state_events: List[Dict] = None
    ):
        """
        Adds a single cropping campaign to the rotation.
        """
        campaign_config = {
            "CropCalendar": {
                "crop_name": crop_name,
                "variety_name": variety_name,
                "crop_start_date": crop_start_date,
                "crop_start_type": crop_start_type,
                "crop_end_date": crop_end_date,
                "crop_end_type": crop_end_type,
                "max_duration": max_duration
            },
            "TimedEvents": timed_events,
            "StateEvents": state_events
        }
        
        # Append the campaign dictionary {start_date: config} to the list
        self.append({campaign_start_date: campaign_config})

    def add_trailing_empty_campaign(self, start_date: datetime.date):
        """
        Adds a final empty campaign to ensure the simulation runs until the very end 
        of the requested period, even after the last crop is harvested.
        """
        self.append({start_date: None})


