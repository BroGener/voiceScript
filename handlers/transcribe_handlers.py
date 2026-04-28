"""
handlers/transcribe_handlers.py
================================
Business logic for the Transcribe & Edit tab.
No Gradio imports — returns plain Python values.
"""

from __future__ import annotations
import sys
import traceback
from pathlib import Path

from config import cfg
from handlers.state import app_state, get_segs, set_segs, get_audio
from handlers.segment_helpers import segs_to_rows, rows_to_segs, TABLE_HEADERS
from speaker_manager import SpeakerManager
from utils import (
    Segment, assign_seg_ids, fmt_srt, load_json,
    save_json, save_srt, derive_output_paths,
)


class _QuietStderr:
    def write(self, *_): pass
    def flush(self): pass


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def run_transcription(audio_file, mode, cut_silence, speaker_filter_str, offset_s):
    """Generator: yields (log_text, table_rows, status_str) progressively."""
    if audio_file is None:
        yield "❌ No audio file selected.", [], "idle"
        return

    audio_path = Path(audio_file if isinstance(audio_file, str) else audio_file.name)
    app_state["audio_path"] = audio_path

    speaker_filter = None
    if speaker_filter_str.strip():
        speaker_filter = [s.strip() for s in speaker_filter_str.split(",") if s.strip()]

    log_lines: list[str] = []

    def log(msg: str):
        log_lines.append(msg)

    orig_stderr = sys.stderr
    sys.stderr   = _QuietStderr()

    try:
        yield "⏳ Starting ...", [], "running"

        from pipeline import run_pipeline
        result = run_pipeline(
            audio_path,
            mode=mode,
            cut_silence=cut_silence,
            speaker_filter=speaker_filter,
            log=log,
        )

        segs = result["segments"]

        # Apply time offset
        try:
            off = float(offset_s)
            if abs(off) > 0.001:
                for seg in segs:
                    seg.start = max(0.0, seg.start + off)
                    seg.end   = max(0.0, seg.end   + off)
        except (ValueError, TypeError):
            pass

        assign_seg_ids(segs, audio_path.stem)
        set_segs(segs)
        app_state["srt_path"]  = result["output_srt"]
        app_state["json_path"] = result["output_json"]

        log_text = "\n".join(log_lines)
        if result.get("pending_mapping"):
            pm = result["pending_mapping"]
            log_text += (f"\n\n⚠️  Unknown speakers need naming.\n"
                         f"   Open: {pm}\n"
                         f"   Fill in real_name, then use Speaker DB tab → Apply Mapping.")

        yield log_text, segs_to_rows(segs), "done ✅"

    except Exception:
        yield f"❌ Error:\n{traceback.format_exc()}", [], "error ❌"
    finally:
        sys.stderr = orig_stderr


# ---------------------------------------------------------------------------
# Load existing transcript
# ---------------------------------------------------------------------------

def load_existing(json_file):
    if json_file is None:
        return "No file selected.", []
    path = Path(json_file if isinstance(json_file, str) else json_file.name)
    try:
        segs = load_json(path)
        stem = path.stem
        for sfx in ("_reconciled", "_whisperx", "_whisper", "_cut"):
            stem = stem.replace(sfx, "")
        assign_seg_ids(segs, stem)
        set_segs(segs)
        app_state["json_path"] = path
        app_state["srt_path"]  = path.with_suffix(".srt")
        return f"✅ Loaded {len(segs)} segments from {path.name}", segs_to_rows(segs)
    except Exception as ex:
        return f"❌ {ex}", []


# ---------------------------------------------------------------------------
# Save & voiceprint update
# ---------------------------------------------------------------------------

def save_segments(rows, confident):
    if not get_segs():
        return "❌ Nothing to save."

    originals  = get_segs()
    corrected  = rows_to_segs(rows, originals)
    set_segs(corrected)

    if app_state["srt_path"]:
        save_srt(corrected, app_state["srt_path"])
    if app_state["json_path"]:
        save_json(corrected, app_state["json_path"])

    diffs = [(o, c) for o, c in zip(originals, corrected) if o.speaker != c.speaker]
    if not diffs:
        return "✅ Saved. No speaker changes."

    audio = get_audio()
    if audio and audio.exists():
        SpeakerManager().apply_corrections_from_srt_diff(
            audio, corrected, originals, confident=confident
        )

    lines = [f"✅ Saved. {len(diffs)} speaker correction(s):"]
    for o, c in diffs[:8]:
        lines.append(f"  [{fmt_srt(o.start)}]  {o.speaker} → {c.speaker}")
    if len(diffs) > 8:
        lines.append(f"  ... and {len(diffs)-8} more")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Time offset
# ---------------------------------------------------------------------------

def apply_time_offset(rows, offset_s):
    try:
        off = float(offset_s)
    except (ValueError, TypeError):
        return rows, "❌ Invalid offset."
    originals = get_segs()
    corrected = rows_to_segs(rows, originals)
    for seg in corrected:
        seg.start = max(0.0, seg.start + off)
        seg.end   = max(0.0, seg.end   + off)
    set_segs(corrected)
    return segs_to_rows(corrected), f"✅ Offset: {off:+.3f}s applied."


# ---------------------------------------------------------------------------
# Voiceprint extraction from selected rows
# ---------------------------------------------------------------------------

def extract_voiceprints(selected_indices_str: str, confident: bool) -> str:
    segs  = get_segs()
    audio = get_audio()
    if not segs or audio is None:
        return "❌ No transcription loaded."

    try:
        indices = [int(x.strip()) for x in selected_indices_str.split(",") if x.strip()]
    except ValueError:
        return "❌ Enter comma-separated row numbers, e.g.  0, 3, 7"

    selected = [(i, segs[i]) for i in indices if 0 <= i < len(segs)]
    if not selected:
        return "❌ No valid row indices."

    try:
        import numpy as np
        from pyannote.audio import Model, Inference
        import torchaudio

        model     = Model.from_pretrained("pyannote/embedding",
                                          use_auth_token=cfg.hf_token)
        inference = Inference(model, window="whole")
        waveform, sr = torchaudio.load(str(audio))
        mgr = SpeakerManager()
        results = []

        for idx, seg in selected:
            if seg.speaker in ("SPEAKER_UNKNOWN", ""):
                results.append(f"  row {idx}: skipped (no speaker label)")
                continue
            chunk = waveform[:, int(seg.start * sr): int(seg.end * sr)]
            try:
                emb    = np.array(inference({"waveform": chunk.unsqueeze(0),
                                              "sample_rate": sr}))
                weight = 0.9 if confident else 0.7
                mgr.store_embeddings(seg.speaker, [emb],
                                     source_file=audio.name,
                                     file_weight=weight)
                results.append(f"  row {idx} [{seg.seg_id}] → {seg.speaker} (w={weight})")
            except Exception as ex:
                results.append(f"  row {idx}: failed — {ex}")

        return "✅ Voiceprint extraction:\n" + "\n".join(results)

    except Exception:
        return f"❌ Error:\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# Export selected segments
# ---------------------------------------------------------------------------

def export_selected(selected_indices_str: str):
    segs = get_segs()
    if not segs:
        return "❌ No transcription loaded.", ""
    try:
        indices = [int(x.strip()) for x in selected_indices_str.split(",") if x.strip()]
    except ValueError:
        return "❌ Invalid indices.", ""

    selected = [segs[i] for i in indices if 0 <= i < len(segs)]
    if not selected:
        return "❌ No valid rows.", ""

    out = cfg.paths.transcripts_dir / f"export_{len(selected)}segs.srt"
    save_srt(selected, out)
    return f"✅ Exported {len(selected)} segments → {out.name}", str(out)


# ---------------------------------------------------------------------------
# Segment-gap silence cutting
# ---------------------------------------------------------------------------

def cut_silence_from_gaps(min_gap_s_str, padding_s_str):
    segs  = get_segs()
    audio = get_audio()
    if not segs or audio is None:
        return "❌ No transcription loaded.", []

    try:
        min_gap = float(min_gap_s_str)
        padding = float(padding_s_str)
    except (ValueError, TypeError):
        return "❌ Invalid numbers.", []

    cut_ranges: list[tuple[float, float]] = []
    for i in range(len(segs) - 1):
        gap_start = segs[i].end
        gap_end   = segs[i + 1].start
        if gap_end - gap_start >= min_gap:
            cut_ranges.append((gap_start, gap_end))

    if not cut_ranges:
        return f"✅ No gaps ≥ {min_gap}s found.", segs_to_rows(segs)

    from audio_processor import AudioProcessor
    proc = AudioProcessor()

    remapped = []
    for seg in segs:
        new_start = proc._remap_time(seg.start, cut_ranges)
        new_end   = proc._remap_time(seg.end,   cut_ranges)
        new_extra = dict(seg.extra)
        new_extra["orig_start"] = seg.start
        new_extra["orig_end"]   = seg.end
        remapped.append(Segment(
            start=new_start, end=new_end,
            text=seg.text, speaker=seg.speaker, source=seg.source,
            seg_id=seg.seg_id,
            text_whisper=seg.text_whisper, text_whisperx=seg.text_whisperx,
            words=seg.words, extra=new_extra,
        ))

    # Cut audio with audible marker
    cut_path    = cfg.paths.processed_audio_dir / f"{audio.stem}_cut.mp3"
    keep_ranges = proc._invert_silence(cut_ranges, audio)
    audio_msg   = ""
    try:
        proc._ffmpeg_concat(audio, keep_ranges, cut_path)
        audio_msg = f"\n🎵 Audio → {cut_path.name}"
    except Exception as ex:
        audio_msg = f"\n⚠️  Audio cut failed: {ex}"

    set_segs(remapped)
    out = derive_output_paths(audio, cfg.paths.transcripts_dir, "_cut")
    save_srt(remapped, out["srt"])
    save_json(remapped, out["json"])
    app_state["srt_path"]  = out["srt"]
    app_state["json_path"] = out["json"]

    return (f"✅ Cut {len(cut_ranges)} gap(s), saved {out['srt'].name}{audio_msg}",
            segs_to_rows(remapped))
