"""
handlers/state.py
=================
Shared application state. All tabs read/write this single dict.
Import: from handlers.state import app_state, update_state
"""

from __future__ import annotations
from pathlib import Path
from utils import Segment

# Central state — mutated in place by all handlers
app_state: dict = {
    "segments":    [],    # list[Segment] — current working segments
    "audio_path":  None,  # Path to original audio
    "srt_path":    None,  # Path to current SRT output
    "json_path":   None,  # Path to current JSON output
}


def get_segs() -> list[Segment]:
    return app_state["segments"]


def set_segs(segs: list[Segment]) -> None:
    app_state["segments"] = segs


def get_audio() -> Path | None:
    return app_state["audio_path"]
