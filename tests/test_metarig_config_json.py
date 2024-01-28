
#%%
import json

from caliscope import __root__
from pathlib import Path

from caliscope.trackers.tracker_enum import TrackerEnum
from caliscope.post_processing.blender_tools import generate_metarig_config

import caliscope.logger
logger = caliscope.logger.get(__name__)
def test_metarig_config_generation():
    xyz_csv_path = Path(__root__,"tests", "reference", "auto_rig_config_data", "xyz_HOLISTIC_OPENSIM_labelled.csv")
    tracker_enum = TrackerEnum.HOLISTIC_OPENSIM
    
    ######## code block for testing purposes  ######
    tracker = tracker_enum.value()
    # for testing purposes, need to make sure that this file is not there before proceeding
    json_path = Path(xyz_csv_path.parent, f"metarig_config_{tracker.name}.json")
    json_path.unlink(missing_ok=True)
    assert not json_path.exists()
    #################################################
    
    
    generate_metarig_config(tracker_enum,xyz_csv_path)


    ######## very basic assertions on generated output  ###
    with open(json_path,"r") as f:
        check_autorig_config = json.load(f)    
  
    # make sure all measures are accounted for and sensible  
    for measure, points in tracker.metarig_symmetrical_measures.items():
        assert measure in check_autorig_config.keys()
        assert type(check_autorig_config[measure]) == float

    for measure, points in tracker.metarig_bilateral_measures.items():
        assert measure in check_autorig_config.keys()
        assert type(check_autorig_config[measure]) == float


if __name__ == "__main__":
    test_metarig_config_generation()