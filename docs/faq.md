
# FAQ


## Can it do real time?

No. The underlying data processing pipeline was constructed with the ultimate goal of real-time tracking, and I have been able to perform some simple tests validating that this can work, but the framerate is sufficiently throttled by the processing demands of the landmark detection across concurrent frames that I don't consider this a priority for development. 

## Why is it only on Windows?



## Can I use my phone as a camera?

No. 

The aim here is an open-source, simple, scaleable, low-cost system. Achieving that means, in part, relying heavily on OpenCV for camera management. Attempts to support input streams that are not directly managed by OpenCV is currently too large of a project in itself and would detract from development and build out of more core processes. 

## What webcams should I buy?

I would suggest you begin with whatever you have immediately on hand and do a test with only 2 (or maybe 3) to understand how things run on your system.  I have had good success with the EMeet HD1080p cameras which are ~$25 on Amazon. Pricier cameras with more features (like autofocus) have created complications in my experience. If you have had success (or frustrations) with a webcam, please let me know so that I can build out a list of cameras that appear to work (or be problematic).

## Can it export to Blender (or Unreal/Maya/etc)?

Not currently. The only formats currently being exported to are `csv` and `trc`. The `trc` format is intended for use by biomechanists. The `csv` file contains unfiltered triangulated landmark point estimates. For someone interested in creating an animation output pipeline, these 'csv' files may be a good place to start. 

## Are you storing my videos?

No. Everything runs locally on your machine. 

## 

