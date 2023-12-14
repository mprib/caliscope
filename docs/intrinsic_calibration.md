# Intrinsic Calibration



- **intrinsic calibration**:  footage for each camera (can be non-synchronized) that has a charuco board moving across the whole FOV of the camera at various distances with the board at various angles to show foreshortening. 

- Important note: If the board is only registered in small parts of the FOV, or is only shown flat relative to the camera, there will be problems in the calibration.  
    

# Extrinsic Calibration
- **extrinsic calibration**: synchronized footage with a charuco board swept throughout the capture volume, ideally touching down to where you would like the origin set 

- "synchronized" here means either all frames are time locked in order, or there is a file that specifies the time at which each frame was recorded which allows the synchronization down stream
- important note: the calibration will improve as the number of concurrent captures of the board from multiple cameras increases."Orphaned" cameras will bungle the optimization if they don't share common views with at least one other camera. 