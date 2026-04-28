"""
handlers/segment_helpers.py
============================
Shared helpers for converting between Segment lists and Gradio table rows.
No Gradio imports here — pure data transformation.
"""

from __future__ import annotations
from utils import Segment, fmt_srt, assign_seg_ids

TABLE_HEADERS = ["#", "ID", "Start", "End", "Speaker", "Text", "Source"]


def segs_to_rows(segments: list[Segment]) -> list[list]:
    return [
        [i, seg.seg_id, fmt_srt(seg.start), fmt_srt(seg.end),
         seg.speaker, seg.text, seg.source]
        for i, seg in enumerate(segments)
    ]


def rows_to_segs(rows: list[list], originals: list[Segment]) -> list[Segment]:
    result = []
    for row, orig in zip(rows, originals):
        result.append(Segment(
            start=orig.start, end=orig.end,
            text=str(row[5]).strip(),
            speaker=str(row[4]).strip(),
            source=orig.source,
            seg_id=orig.seg_id,
            text_whisper=orig.text_whisper,
            text_whisperx=orig.text_whisperx,
            words=orig.words,
            extra=orig.extra,
        ))
    return result


def conflict_rows(segments: list[Segment]) -> list[list]:
    """Return only conflict segments formatted for the review table."""
    return [
        [i,
         fmt_srt(seg.start),
         fmt_srt(seg.end),
         seg.speaker,
         seg.text_whisper,
         seg.text_whisperx]
        for i, seg in enumerate(segments)
        if seg.source == "both"
    ]
