"""
handlers/review_handlers.py
============================
Business logic for the Review tab (conflict resolution + audio player).
"""

from __future__ import annotations
from pathlib import Path

from handlers.state import get_segs, set_segs, get_audio
from handlers.segment_helpers import conflict_rows
from utils import Segment, fmt_srt, parse_srt_time, save_srt, save_json
from config import cfg


# ---------------------------------------------------------------------------
# Conflict loading
# ---------------------------------------------------------------------------

def load_conflicts():
    segs = get_segs()
    rows = conflict_rows(segs)
    if not rows:
        return "✅ No conflicts in current transcript.", []
    return f"Found {len(rows)} conflict(s). Click a row to resolve.", rows


# ---------------------------------------------------------------------------
# Resolve single conflict
# ---------------------------------------------------------------------------

def resolve_one(seg_idx_str: str, choice: str, speaker_override: str):
    """
    Resolve a single conflict segment.

    choice:           "whisper" | "whisperx" | "keep both"
    speaker_override: if non-empty, overrides the speaker label
    Returns (status_msg, updated_conflict_rows, html_diff)
    """
    segs = get_segs()
    try:
        idx = int(seg_idx_str)
    except ValueError:
        return "❌ Enter a valid segment number.", conflict_rows(segs), ""

    if idx >= len(segs) or segs[idx].source != "both":
        return "❌ Not a conflict segment.", conflict_rows(segs), ""

    seg = segs[idx]
    speaker = speaker_override.strip() or seg.speaker

    if choice == "whisper":
        text = seg.text_whisper
        src  = "whisper"
    elif choice == "whisperx":
        text = seg.text_whisperx
        src  = "whisperx"
    else:
        # keep both — just update speaker if overridden
        segs[idx] = Segment(
            start=seg.start, end=seg.end, text=seg.text,
            speaker=speaker, source="both", seg_id=seg.seg_id,
            text_whisper=seg.text_whisper, text_whisperx=seg.text_whisperx,
            words=seg.words, extra=seg.extra,
        )
        set_segs(segs)
        return f"Row {idx}: kept both versions.", conflict_rows(segs), ""

    segs[idx] = Segment(
        start=seg.start, end=seg.end, text=text,
        speaker=speaker, source=src, seg_id=seg.seg_id,
        words=seg.words, extra=seg.extra,
    )
    set_segs(segs)
    return f"✅ Row {idx}: kept {choice} → '{text[:60]}'", conflict_rows(segs), ""


# ---------------------------------------------------------------------------
# Resolve all conflicts at once
# ---------------------------------------------------------------------------

def resolve_all(choice: str) -> str:
    segs  = get_segs()
    count = 0
    for i, seg in enumerate(segs):
        if seg.source != "both":
            continue
        text = seg.text_whisper if choice == "whisper" else seg.text_whisperx
        segs[i] = Segment(
            start=seg.start, end=seg.end, text=text,
            speaker=seg.speaker, source=choice, seg_id=seg.seg_id,
            words=seg.words, extra=seg.extra,
        )
        count += 1

    set_segs(segs)
    from config import cfg
    from handlers.state import app_state
    if app_state.get("srt_path"):
        save_srt(segs, app_state["srt_path"])
    if app_state.get("json_path"):
        save_json(segs, app_state["json_path"])
    return f"✅ Resolved {count} conflict(s) → kept {choice} for all."


# ---------------------------------------------------------------------------
# Split a segment at a word boundary (manual merge-repair)
# STUB — UI placeholder; full implementation after clustering improvements
# ---------------------------------------------------------------------------

def split_segment(seg_idx_str: str, split_word_idx_str: str, speaker_b: str):
    """
    Split segment at word N into two segments, assigning speaker_b to the second.
    Requires word-level alignment data in seg.words.
    Returns (status, updated_conflict_rows)
    """
    segs = get_segs()
    try:
        idx        = int(seg_idx_str)
        word_idx   = int(split_word_idx_str)
    except ValueError:
        return "❌ Enter valid numbers.", conflict_rows(segs)

    if idx >= len(segs):
        return "❌ Invalid segment index.", conflict_rows(segs)

    seg = segs[idx]
    words = seg.words

    if not words:
        return "⚠️  No word-level data available for this segment (alignment missing).", conflict_rows(segs)

    if word_idx <= 0 or word_idx >= len(words):
        return f"❌ Word index must be 1–{len(words)-1}.", conflict_rows(segs)

    # Split words
    words_a = words[:word_idx]
    words_b = words[word_idx:]

    def _word_time(w, key):
        # whisperx word dicts use 'start'/'end' keys
        return float(w.get(key, 0.0))

    text_a = " ".join(w.get("word", "") for w in words_a).strip()
    text_b = " ".join(w.get("word", "") for w in words_b).strip()
    end_a  = _word_time(words_a[-1], "end") if words_a else seg.start
    start_b = _word_time(words_b[0], "start") if words_b else end_a

    from utils import make_seg_id
    stem = get_audio().stem if get_audio() else "unknown"

    seg_a = Segment(
        start=seg.start, end=end_a, text=text_a,
        speaker=seg.speaker, source=seg.source, seg_id=seg.seg_id,
        words=words_a, extra=seg.extra,
    )
    seg_b = Segment(
        start=start_b, end=seg.end, text=text_b,
        speaker=speaker_b.strip() or "SPEAKER_UNKNOWN",
        source=seg.source,
        seg_id=make_seg_id(stem, start_b),
        words=words_b, extra={},
    )

    # Replace original with two segments
    segs = segs[:idx] + [seg_a, seg_b] + segs[idx+1:]
    segs.sort(key=lambda s: s.start)
    set_segs(segs)

    return (f"✅ Split row {idx} at word {word_idx}. "
            f"'{text_a[:40]}' | '{text_b[:40]}'",
            conflict_rows(segs))


# ---------------------------------------------------------------------------
# Diff HTML for side-by-side display
# ---------------------------------------------------------------------------

def make_diff_html(seg_idx_str: str) -> str:
    """
    Generate a side-by-side HTML diff for the selected conflict segment.
    Left = Whisper (orange), Right = WhisperX (blue).
    """
    segs = get_segs()
    try:
        idx = int(seg_idx_str)
    except ValueError:
        return ""

    if idx >= len(segs) or segs[idx].source != "both":
        return "<p style='color:gray'>Select a conflict row to see diff.</p>"

    seg = segs[idx]

    def _hi(text: str, color: str) -> str:
        return (f"<div style='background:{color};padding:12px;border-radius:6px;"
                f"font-family:monospace;font-size:13px;line-height:1.6'>{text}</div>")

    left  = _hi(seg.text_whisper  or "(empty)", "#fff3e0")   # warm orange tint
    right = _hi(seg.text_whisperx or "(empty)", "#e3f2fd")   # cool blue tint

    words_info = ""
    if seg.words:
        words_info = (f"<p style='font-size:11px;color:#888;margin-top:4px'>"
                      f"Word-level data available: {len(seg.words)} words "
                      f"(can be split)</p>")

    return f"""
    <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px'>
      <div>
        <div style='font-weight:bold;margin-bottom:4px;color:#e65100'>
          🎤 Whisper
        </div>
        {left}
      </div>
      <div>
        <div style='font-weight:bold;margin-bottom:4px;color:#1565c0'>
          🎤 WhisperX
        </div>
        {right}
      </div>
    </div>
    <div style='margin-top:8px;font-size:12px;color:#555'>
      Speaker: <b>{seg.speaker}</b> &nbsp;|&nbsp;
      {fmt_srt(seg.start)} → {fmt_srt(seg.end)} &nbsp;|&nbsp;
      ID: {seg.seg_id}
    </div>
    {words_info}
    """


# ---------------------------------------------------------------------------
# Player: seek to segment
# ---------------------------------------------------------------------------

def seek_to_segment(evt_data, audio_path_str: str):
    """
    Called on Dataframe row select. Seeks audio to segment start time.
    Returns (audio_value, status_str)
    gr.Audio accepts (filepath, start_time) tuple for seeking.
    """
    if not audio_path_str:
        return None, "No audio loaded."
    try:
        # evt_data is the selected row list from gr.Dataframe
        if isinstance(evt_data, list) and len(evt_data) >= 2:
            start_str = str(evt_data[1])      # Start column
        else:
            return None, "Could not read row data."
        start_s = parse_srt_time(start_str)
        return (audio_path_str, start_s), f"⏩ {start_str}"
    except Exception as ex:
        return None, f"❌ {ex}"


def get_player_rows():
    segs = get_segs()
    return [
        [fmt_srt(seg.start), fmt_srt(seg.end), seg.speaker, seg.text[:80]]
        for seg in segs
    ]
