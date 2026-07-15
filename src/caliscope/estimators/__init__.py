"""Estimators — extractors that eat video and emit observation types.

Parallels ``trackers/``: trackers emit 2D landmarks (ImagePoints), estimators
emit metric cues (focal lengths, depths, vertical directions) consumed by the
anchoring API. Import runners from their modules directly, e.g.
``caliscope.estimators.moge`` or ``caliscope.estimators.vertical``.
"""
