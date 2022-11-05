# Overview

Run `src\gui\main.py` to launch a dialog for individual camera calibration. This will update parameters to `test_session\config.toml`. When intrinsic parameters are set, then launch `src\calibration\stereocalibrator.py`. This will open up an opencv window with the stacked frame pairs for stereocalibration.

Within almost all the primary code modules there should be an `if __name__ == "__main__":` showcase at the bottom. No proper tests yet, but this is where someone might look to get a quick and dirty idea of what is going on. My immediate goals are to improve logging, get the stereocalibration into the dialog, and get a way to visualize the stereocalibration results in 3D.

A walk through of the current functionality is here:

https://youtu.be/64-Lv390SMo

