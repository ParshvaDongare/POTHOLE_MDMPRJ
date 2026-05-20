"""Compatibility shim for the latest pothole pipeline.

The project's canonical implementation now lives in `pothole_detection_pipeline_v3`.
This module re-exports that API so existing imports keep working.
"""

from pothole_detection_pipeline_v3 import *  # noqa: F401,F403
