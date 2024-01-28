import rtoml
from pathlib import Path
import caliscope.logger
from caliscope.tracker import WireFrameView, Segment
logger = caliscope.logger.get(__name__)

def get_wireframe(toml_spec_path: Path, point_names: dict)-> WireFrameView:
    
    # load in toml 
    wireframe_specs = rtoml.load(toml_spec_path)

    logger.info(f"Building following wireframe: {wireframe_specs}")

    # build out a list of segments based on the dictionary
    segments = []
    
    for segment_name, specs in wireframe_specs.items():
        segment = Segment(name=segment_name,
                          color=specs['color'],
                          point_A=specs['points'][0],
                          point_B=specs['points'][1],
                          )
        segments.append(segment)

    wireframe = WireFrameView(segments=segments,point_names=point_names)        

    return wireframe