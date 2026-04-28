"""
utils.py
========
Segment dataclass, time formatting, SRT/JSON output.

Each Segment gets a stable seg_id (8-char hex) derived from the audio stem
and start time. This ID is used to:
  - Link voiceprint embeddings back to the exact segment they came from
  - Identify segments across edits (time may change, ID stays the same)
  - Support selective export and cut operations in the web UI
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# seg_id generation
# ---------------------------------------------------------------------------

def make_seg_id(audio_stem: str, start: float) -> str:
    """
    Generate a stable 8-char hex ID for a segment.
    ID = SHA1(audio_stem + start_ms)[:8]
    Stable across runs as long as the audio file and start time don't change.
    """
    key = f"{audio_stem}:{int(round(start * 1000))}"
    return hashlib.sha1(key.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Segment
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """One transcript segment with full metadata."""
    start: float
    end: float
    text: str
    speaker: str = "SPEAKER_UNKNOWN"
    source: Literal["whisper", "whisperx", "reconciled", "both"] = "reconciled"
    # Kept when source == "both"
    text_whisper: str = ""
    text_whisperx: str = ""
    # Word-level alignment from whisperx align
    words: list = field(default_factory=list)
    # Stable identifier — set by make_seg_id(), survives edits
    seg_id: str = ""
    # Extensible metadata: orig_start/end for remapped segments, confidence, etc.
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def assign_id(self, audio_stem: str) -> "Segment":
        """Set seg_id if not already assigned. Returns self for chaining."""
        if not self.seg_id:
            self.seg_id = make_seg_id(audio_stem, self.start)
        return self


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------

def fmt_srt(seconds: float) -> str:
    """Seconds → SRT timestamp  HH:MM:SS,mmm"""
    ms = int(round((seconds % 1) * 1000))
    s  = int(seconds) % 60
    m  = (int(seconds) // 60) % 60
    h  = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_srt_time(ts: str) -> float:
    """SRT timestamp string → seconds (inverse of fmt_srt)."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + s


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def save_srt(segments: list[Segment], path: Path) -> None:
    """
    Write standard SRT subtitle file.
    Conflict segments (source='both') get two lines: [WHISPER] and [WHISPERX].
    Segments with orig_start/orig_end in extra get an [orig: ...] annotation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, 1):
            time_line = f"{fmt_srt(seg.start)} --> {fmt_srt(seg.end)}"
            orig_note = _orig_time_note(seg)
            id_note   = f" #{seg.seg_id}" if seg.seg_id else ""

            if seg.source == "both":
                f.write(
                    f"{idx}\n{time_line}\n"
                    f"  ({seg.speaker}){id_note} [WHISPER]  : {seg.text_whisper}{orig_note}\n"
                    f"  ({seg.speaker}) [WHISPERX] : {seg.text_whisperx}\n\n"
                )
            else:
                src_tag = f"[{seg.source.upper()}] " if seg.source != "reconciled" else ""
                f.write(
                    f"{idx}\n{time_line}\n"
                    f"  ({seg.speaker}){id_note} {src_tag}{seg.text}{orig_note}\n\n"
                )
    print(f"  💾 SRT  → {path}")


def save_json(segments: list[Segment], path: Path) -> None:
    """Write full JSON (includes word-level alignment and all extra fields)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in segments], f, indent=2, ensure_ascii=False)
    print(f"  💾 JSON → {path}")


def load_json(path: Path) -> list[Segment]:
    """Load Segment list from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segs = []
    for d in data:
        # Backwards compat: old JSONs without seg_id
        d.setdefault("seg_id", "")
        segs.append(Segment(**d))
    return segs


def derive_output_paths(audio_path: Path, out_dir: Path, suffix: str = "") -> dict[str, Path]:
    """Derive SRT/JSON output paths from audio filename."""
    stem = audio_path.stem + suffix
    return {
        "srt":  out_dir / f"{stem}.srt",
        "json": out_dir / f"{stem}.json",
    }


def assign_seg_ids(segments: list[Segment], audio_stem: str) -> list[Segment]:
    """Assign seg_ids to all segments that don't have one yet."""
    for seg in segments:
        seg.assign_id(audio_stem)
    return segments


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _orig_time_note(seg: Segment) -> str:
    orig_start = seg.extra.get("orig_start")
    orig_end   = seg.extra.get("orig_end")
    if orig_start is not None and orig_end is not None:
        return f"  [orig: {fmt_srt(orig_start)} --> {fmt_srt(orig_end)}]"
    return ""
