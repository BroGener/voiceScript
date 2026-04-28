"""
pipeline.py
===========
Core transcription pipeline — shared by main.py (CLI) and gradio_app.py (web).

All heavy lifting lives here. Entry points just call run_pipeline().
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Callable, Iterator

from config import cfg
from reconciler import Reconciler
from speaker_manager import SpeakerManager
from transcriber_whisper import WhisperTranscriber
from transcriber_whisperx import WhisperXTranscriber
from utils import derive_output_paths, save_json, save_srt


# ---------------------------------------------------------------------------
# Log callback type: pipeline emits log lines, caller decides where they go
# (print for CLI, gradio yield for web)
# ---------------------------------------------------------------------------

LogFn = Callable[[str], None]


def _noop(msg: str) -> None:
    print(msg)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    audio_path: Path | str,
    *,
    mode: str = "dual",                  # "dual" | "whisper" | "whisperx"
    cut_silence: bool = False,           # apply silence cutting before transcribe
    speaker_filter: list[str] | None = None,  # limit matching to these speakers
    log: LogFn = _noop,
) -> dict:
    """
    Run the full transcription pipeline.

    Parameters
    ----------
    audio_path     : input audio file
    mode           : "dual" (Whisper + WhisperX + reconcile)
                     "whisper" (Whisper only)
                     "whisperx" (WhisperX only)
    cut_silence    : if True, cut silence first (remap mode — no re-transcription)
    speaker_filter : list of known speaker names to restrict matching to;
                     None = use all speakers in database
    log            : callable for progress messages

    Returns
    -------
    {
        "segments":    list[Segment],   # final merged/selected segments
        "output_srt":  Path,
        "output_json": Path,
        "pending_mapping": Path | None, # set if cold-start produced a mapping file
    }
    """
    audio_path = Path(audio_path)
    out_dir    = cfg.paths.transcripts_dir

    # ── Optional silence cutting (remap mode, no re-transcription) ──────────
    if cut_silence:
        log("✂️  Cutting silence (remap mode) ...")
        from audio_processor import AudioProcessor
        proc = AudioProcessor()
        cfg.audio_processor.re_transcribe_after_cut = False
        cut_path, _ = proc.cut_silence(audio_path)
        if cut_path != audio_path:
            log(f"   Cut audio → {cut_path.name}")
            # Load existing segments for remapping if available
            existing_json = out_dir / f"{audio_path.stem}_reconciled.json"
            if existing_json.exists():
                from utils import load_json
                existing_segs = load_json(existing_json)
                silence_ranges = proc._detect_silence(audio_path)
                remapped = proc._remap_segments(existing_segs, silence_ranges)
                from utils import fmt_srt
                for seg in remapped:
                    seg.extra["orig_start"] = seg.extra.get("orig_start", seg.start)
                    seg.extra["orig_end"]   = seg.extra.get("orig_end",   seg.end)
                out_paths = derive_output_paths(audio_path, out_dir, "_cut")
                save_srt(remapped, out_paths["srt"])
                save_json(remapped, out_paths["json"])
                log(f"   Remapped {len(remapped)} segments → {out_paths['srt'].name}")
                return {
                    "segments":       remapped,
                    "output_srt":     out_paths["srt"],
                    "output_json":    out_paths["json"],
                    "pending_mapping": None,
                }
            else:
                log("   No existing transcription to remap — will transcribe cut audio.")
                audio_path = cut_path

    # ── Speaker manager (apply filter if requested) ──────────────────────────
    speaker_mgr = SpeakerManager()
    if speaker_filter:
        speaker_mgr = _filtered_speaker_manager(speaker_mgr, speaker_filter)
        log(f"🔍 Speaker filter: {speaker_filter}")
    else:
        if speaker_mgr.has_speakers():
            log(f"👥 Known speakers: {speaker_mgr.list_speakers()}")
        else:
            log("👥 No speaker database yet — cold-start mode.")

    pending_mapping: Path | None = None

    # ── Mode: Whisper only ───────────────────────────────────────────────────
    if mode == "whisper":
        log("\n🎙️  Whisper transcription ...")
        wt   = WhisperTranscriber()
        segs = wt.transcribe(audio_path)
        wt.unload()
        p = derive_output_paths(audio_path, out_dir, "_whisper")
        save_srt(segs, p["srt"])
        save_json(segs, p["json"])
        log(f"✅ Done: {len(segs)} segments → {p['srt'].name}")
        return {"segments": segs, "output_srt": p["srt"],
                "output_json": p["json"], "pending_mapping": None}

    # ── Mode: WhisperX only ─────────────────────────────────────────────────
    if mode == "whisperx":
        log("\n🎙️  WhisperX transcription ...")
        wxt  = WhisperXTranscriber()
        segs = wxt.transcribe(audio_path, speaker_manager=speaker_mgr)
        wxt.unload()
        p = derive_output_paths(audio_path, out_dir, "_whisperx")
        save_srt(segs, p["srt"])
        save_json(segs, p["json"])
        pending_files = list(cfg.paths.pending_mapping_dir.glob('pending_*.json'))
        if pending_files:
            pending_mapping = pending_files[0]  # most recent cold-start
        log(f"✅ Done: {len(segs)} segments → {p['srt'].name}")
        return {"segments": segs, "output_srt": p["srt"],
                "output_json": p["json"], "pending_mapping": pending_mapping}

    # ── Mode: Dual (default) ─────────────────────────────────────────────────
    log("\n🎙️  Whisper transcription ...")
    wt     = WhisperTranscriber()
    w_segs = wt.transcribe(audio_path)
    wt.unload()
    wp = derive_output_paths(audio_path, out_dir, "_whisper")
    save_srt(w_segs, wp["srt"])
    save_json(w_segs, wp["json"])
    log(f"   Whisper: {len(w_segs)} segments")

    log("\n🎙️  WhisperX transcription ...")
    wxt     = WhisperXTranscriber()
    wx_segs = wxt.transcribe(audio_path, speaker_manager=speaker_mgr)
    wxt.unload()
    wxp = derive_output_paths(audio_path, out_dir, "_whisperx")
    save_srt(wx_segs, wxp["srt"])
    save_json(wx_segs, wxp["json"])
    log(f"   WhisperX: {len(wx_segs)} segments")

    pending_files = list(cfg.paths.pending_mapping_dir.glob("pending_*.json"))
    if pending_files:
        pending_mapping = pending_files[0]

    log("\n🔀 Reconciling ...")
    merged = Reconciler().reconcile(w_segs, wx_segs)
    mp = derive_output_paths(audio_path, out_dir, "_reconciled")
    save_srt(merged, mp["srt"])
    save_json(merged, mp["json"])

    gc.collect()
    log(f"✅ Done: {len(merged)} segments → {mp['srt'].name}")

    return {
        "segments":       merged,
        "output_srt":     mp["srt"],
        "output_json":    mp["json"],
        "pending_mapping": pending_mapping,
    }


# ---------------------------------------------------------------------------
# Speaker filter helper
# ---------------------------------------------------------------------------

class _FilteredSpeakerManager:
    """Wraps SpeakerManager but restricts matching to a subset of speakers."""

    def __init__(self, mgr: SpeakerManager, allowed: list[str]) -> None:
        self._mgr     = mgr
        self._allowed = set(allowed)

    def has_speakers(self) -> bool:
        return any(n in self._allowed for n in self._mgr.list_speakers())

    def list_speakers(self) -> list[str]:
        return [n for n in self._mgr.list_speakers() if n in self._allowed]

    def all_centroids(self) -> dict:
        return {k: v for k, v in self._mgr.all_centroids().items()
                if k in self._allowed}

    def match_speaker(self, embedding):
        import numpy as np
        from speaker_manager import _cos
        centroids = self.all_centroids()
        if not centroids:
            return None, 0.0
        best_name, best_score = None, -1.0
        for name, c in centroids.items():
            score = _cos(embedding, c)
            if score > best_score:
                best_score = score
                best_name  = name
        min_conf = cfg.speaker.remap_min_confidence
        return (best_name, best_score) if best_score >= min_conf else (None, best_score)

    # Forward everything else to the real manager
    def __getattr__(self, name):
        return getattr(self._mgr, name)


def _filtered_speaker_manager(mgr: SpeakerManager, allowed: list[str]):
    return _FilteredSpeakerManager(mgr, allowed)
