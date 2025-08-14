This test data relates to the issue here: https://github.com/mprib/caliscope/issues/688

The user reports the following log messages though we have both confirmed that there are actually shared points:

     INFO| caliscope.calibration.stereocalibrator| 242|  Deleting previous stereocalibrations from config
     INFO| caliscope.calibration.stereocalibrator| 248|  Beginning stereocalibration of pairs [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
     INFO| caliscope.calibration.stereocalibrator| 170|  Assembling 10 shared boards for pair (1, 2)
     INFO| caliscope.calibration.stereocalibrator| 316|  RMSE of reprojection for pair (1, 2) is 1.0746374045284361
     INFO| caliscope.calibration.stereocalibrator| 152|  For pair (1, 3) there are no shared points
     INFO| caliscope.calibration.stereocalibrator| 319|  No stereocalibration produced for pair (1, 3)
     INFO| caliscope.calibration.stereocalibrator| 152|  For pair (1, 4) there are no shared points
     INFO| caliscope.calibration.stereocalibrator| 319|  No stereocalibration produced for pair (1, 4)
     INFO| caliscope.calibration.stereocalibrator| 170|  Assembling 10 shared boards for pair (2, 3)
     INFO| caliscope.calibration.stereocalibrator| 316|  RMSE of reprojection for pair (2, 3) is 1.0947902449310738
     INFO| caliscope.calibration.stereocalibrator| 152|  For pair (2, 4) there are no shared points
     INFO| caliscope.calibration.stereocalibrator| 319|  No stereocalibration produced for pair (2, 4)
     INFO| caliscope.calibration.stereocalibrator| 152|  For pair (3, 4) there are no shared points
     INFO| caliscope.calibration.stereocalibrator| 319|  No stereocalibration produced for pair (3, 4)
     INFO| caliscope.calibration.stereocalibrator| 266|  Direct stereocalibration complete for all pairs for which data is available
