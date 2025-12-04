"""Module to prepare crop parameters"""

import yaml
from pcse.input import YAMLCropDataProvider
import importlib.resources as pkg_resources
from .configs import wofost_crop_params

class WOFOSTCropParametersProvider(YAMLCropDataProvider):
    def __init__(self, crop_name, variety_name):
        # 1. Get the directory path directly from the module
        config_path = list(wofost_crop_params.__path__)[0]
        
        # 2. Initialize the parent class with the string path
        super().__init__(fpath=str(config_path))
        
        # 3. Set the active crop
        self.set_active_crop(crop_name, variety_name)
        self.crop_name = crop_name
        self.variety_name = variety_name
        self.param_metadata = self._get_param_metadata()
        
    def _get_param_metadata(self):
        with pkg_resources.files(wofost_crop_params).joinpath(f"{self.crop_name}.yaml").open("r") as f:
            crop_config = yaml.safe_load(f)
            crop_variety_config = crop_config['CropParameters']['Varieties'][self.variety_name]
            
            param_metadata = []
            for param, info in crop_variety_config.items():
                try:
                    param_dict = {
                        'parameter': param,
                        'description': info[1],
                        'unit': info[-1][0] if len(info[-1]) == 1 else info[-1],
                        'default': info[0]
                    }
                    param_metadata.append(param_dict)
                except:
                    continue
            
            return param_metadata
        
    
def get_available_crops(model):
    if model.startswith("Wofost"):
        with pkg_resources.files(wofost_crop_params).joinpath("crops.yaml").open("r") as f:
            available_crops = yaml.safe_load(f)['available_crops']

        return available_crops
    
    else:
        return []
    
def get_available_crop_varieties(model, crop):
    if model.startswith("Wofost"):
        with pkg_resources.files(wofost_crop_params).joinpath(f"{crop}.yaml").open("r") as f:
            crop_config = yaml.safe_load(f)
            
            all_crop_varieties = crop_config['CropParameters']['Varieties'].keys()
            crop_varieties = {}
            
            for v in all_crop_varieties:
                meta = crop_config['CropParameters']['Varieties'][v]['Metadata']
                crop_varieties[v] = meta
                
        return crop_varieties

    else:
        return None
            
    