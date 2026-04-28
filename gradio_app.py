"""
gradio_app.py — Gradio web interface for Whisper Suite.

Launch via:  python main.py --web
         or: python gradio_app.py

Features:
  - File browser for audio selection
  - Mode selector (dual / whisper / whisperx)
  - Speaker filter (restrict to known speakers)
  - Silence cutting toggle
  - Real-time log streaming
  - Editable segment table with speaker dropdown
  - "Save & update voiceprints" button
  - Speaker database management tab
"""

from __future__ import annotations

import io
import json
import sys
import threading
from pathlib import Path
from typing import Generator

import gradio as gr

from config import cfg
from speaker_manager import SpeakerManager
from utils import Segment, load_json, save_json, save_srt


# ---------------------------------------------------------------------------
# Suppress noisy stderr from whisperx / faster-whisper / ctranslate2
# ---------------------------------------------------------------------------

class _SuppressStderr:
    """Context manager that redirects stderr to /dev/null."""
    def __enter__(self):
        self._orig = sys.stderr
        sys.stderr = open(os.devnull, "w") if hasattr(sys, "stderr") else sys.stderr
        return self
    def __exit__(self, *_):
        try:
            sys.stderr.close()
        except Exception:
            pass
        sys.stderr = self._orig

import os


# ---------------------------------------------------------------------------
# Log capture: redirect print() to a string buffer for Gradio streaming
# ---------------------------------------------------------------------------

class _LogCapture:
    def __init__(self):
        self._buf  = io.StringIO()
        self._lock = threading.Lock()

    def write(self, msg: str) -> None:
        with self._lock:
            self._buf.write(msg)

    def flush(self) -> None:
        pass

    def getvalue(self) -> str:
        with self._lock:
            return self._buf.getvalue()

    def __enter__(self):
        self._orig_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, *_):
        sys.stdout = self._orig_stdout


# ---------------------------------------------------------------------------
# Segment table helpers
# ---------------------------------------------------------------------------

def _segs_to_table(segments: list[Segment]) -> list[list]:
    """Convert segments to a list-of-lists for gr.Dataframe."""
    rows = []
    for i, seg in enumerate(segments):
        from utils import fmt_srt
        rows.append([
            i,
            fmt_srt(seg.start),
            fmt_srt(seg.end),
            seg.speaker,
            seg.text,
            seg.source,
        ])
    return rows


def _table_to_segs(
    rows: list[list],
    original_segs: list[Segment],
) -> list[Segment]:
    """Reconstruct Segment list from edited table rows."""
    result = []
    for row, orig in zip(rows, original_segs):
        # row: [idx, start_str, end_str, speaker, text, source]
        result.append(Segment(
            start=orig.start,
            end=orig.end,
            text=str(row[4]).strip(),
            speaker=str(row[3]).strip(),
            source=orig.source,
            text_whisper=orig.text_whisper,
            text_whisperx=orig.text_whisperx,
            words=orig.words,
            extra=orig.extra,
        ))
    return result


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

_state: dict = {
    "segments":   [],    # list[Segment] — current working segments
    "audio_path": None,  # Path
    "json_path":  None,  # Path
    "srt_path":   None,  # Path
}


def _get_known_speakers() -> list[str]:
    return SpeakerManager().list_speakers()


def _speaker_choices() -> list[str]:
    return ["(unknown / new)"] + _get_known_speakers()


# ---------------------------------------------------------------------------
# Core pipeline runner (streaming logs to Gradio)
# ---------------------------------------------------------------------------

def run_transcription(
    audio_file,
    mode: str,
    cut_silence: bool,
    speaker_filter_str: str,
) -> Generator[tuple, None, None]:
    """
    Gradio generator function — yields (log_text, table_data, status) progressively.
    """
    if audio_file is None:
        yield "❌ No audio file selected.", [], "idle"
        return

    audio_path = Path(audio_file.name if hasattr(audio_file, "name") else audio_file)
    _state["audio_path"] = audio_path

    speaker_filter = None
    if speaker_filter_str.strip():
        speaker_filter = [s.strip() for s in speaker_filter_str.split(",") if s.strip()]

    log_cap   = _LogCapture()
    log_lines = []

    def log(msg: str) -> None:
        log_lines.append(msg)

    # Run in same thread (Gradio generator keeps UI responsive via yield)
    with log_cap:
        import patches
        patches.apply_all()

        from pipeline import run_pipeline

        try:
            yield "\n".join(log_lines) + "\n⏳ Starting ...", [], "running"

            result = run_pipeline(
                audio_path,
                mode=mode,
                cut_silence=cut_silence,
                speaker_filter=speaker_filter,
                log=log,
            )

            segs = result["segments"]
            _state["segments"]  = segs
            _state["srt_path"]  = result["output_srt"]
            _state["json_path"] = result["output_json"]

            table = _segs_to_table(segs)
            final_log = "\n".join(log_lines)

            if result.get("pending_mapping"):
                pm = result["pending_mapping"]
                final_log += (
                    f"\n\n⚠️  Cold-start: fill in speaker names in:\n"
                    f"   {pm}\n"
                    f"   Then use the 'Speaker Setup' tab to apply."
                )

            yield final_log, table, "done"

        except Exception as ex:
            import traceback
            err = traceback.format_exc()
            yield f"❌ Error:\n{err}", [], "error"


# ---------------------------------------------------------------------------
# Save & update voiceprints
# ---------------------------------------------------------------------------

def save_and_update(
    table_rows: list[list],
    confident: bool,
) -> str:
    if not _state["segments"] or _state["audio_path"] is None:
        return "❌ No transcription loaded."

    original_segs  = _state["segments"]
    corrected_segs = _table_to_segs(table_rows, original_segs)

    # Find changed speakers
    diffs = [
        (orig, corr) for orig, corr in zip(original_segs, corrected_segs)
        if orig.speaker != corr.speaker
    ]

    # Save updated SRT and JSON
    if _state["srt_path"]:
        save_srt(corrected_segs, _state["srt_path"])
    if _state["json_path"]:
        save_json(corrected_segs, _state["json_path"])

    _state["segments"] = corrected_segs

    if not diffs:
        return "✅ Saved. No speaker changes detected — voiceprint DB unchanged."

    # Update voiceprint database
    mgr = SpeakerManager()
    mgr.apply_corrections_from_srt_diff(
        _state["audio_path"],
        corrected_segs,
        original_segs,
        confident=confident,
    )

    lines = [f"✅ Saved. {len(diffs)} speaker correction(s) applied:"]
    for orig, corr in diffs[:10]:
        from utils import fmt_srt
        lines.append(f"  [{fmt_srt(orig.start)}] {orig.speaker} → {corr.speaker}")
    if len(diffs) > 10:
        lines.append(f"  ... and {len(diffs)-10} more")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Speaker DB tab helpers
# ---------------------------------------------------------------------------

def db_summary() -> str:
    mgr = SpeakerManager()
    if not mgr.has_speakers():
        return "Database is empty."
    lines = []
    for name in mgr.list_speakers():
        data  = mgr._db[name]
        lines.append(
            f"  {name:<20}  "
            f"anchors={len(data.get('anchors',[]))}  "
            f"embeddings={len(data.get('embeddings',[]))}  "
            f"soft={len(data.get('soft_corrections',[]))}  "
            f"corrections={data.get('correction_count',0)}"
        )
    return "\n".join(lines)


def apply_pending_mapping(mapping_file, audio_file_for_mapping, as_anchor: bool) -> str:
    if mapping_file is None:
        return "❌ No mapping file selected."
    mapping_path = Path(mapping_file.name)
    mapping_data = json.loads(mapping_path.read_text(encoding="utf-8"))
    speakers     = mapping_data.get("speakers", {})
    unfilled     = [k for k, v in speakers.items() if not v.get("real_name", "").strip()]
    if unfilled:
        return f"❌ These labels have no real_name: {unfilled}\nEdit the JSON file first."

    label_to_name = {k: v["real_name"].strip() for k, v in speakers.items()}
    source_file   = mapping_data.get("source_file", "")

    # Rename labels in SRT/JSON transcripts
    from speaker_setup import _rename_labels_in_srt, _rename_labels_in_json
    updated = []
    for srt_p in cfg.paths.transcripts_dir.glob(f"{Path(source_file).stem}*.srt"):
        _rename_labels_in_srt(srt_p, label_to_name)
        updated.append(srt_p.name)
    for json_p in cfg.paths.transcripts_dir.glob(f"{Path(source_file).stem}*.json"):
        _rename_labels_in_json(json_p, label_to_name)
        updated.append(json_p.name)

    # Seed database if audio provided
    db_msg = ""
    if audio_file_for_mapping is not None:
        audio_p = Path(audio_file_for_mapping.name)
        from speaker_setup import _seed_database
        _seed_database(audio_p, mapping_data, label_to_name, as_anchor=as_anchor)
        db_msg = f"\nDatabase seeded from {audio_p.name}."

    mapping_path.unlink(missing_ok=True)
    return (f"✅ Applied mapping: {label_to_name}\n"
            f"Updated files: {', '.join(updated)}{db_msg}")


def set_file_weight_ui(filename: str, weight: float) -> str:
    if not filename.strip():
        return "❌ Enter a filename."
    SpeakerManager().set_file_weight(filename.strip(), weight)
    return f"✅ Weight set to {weight} for all embeddings from '{filename}'."


def drop_file_ui(filename: str) -> str:
    if not filename.strip():
        return "❌ Enter a filename."
    SpeakerManager().drop_file_embeddings(filename.strip())
    return f"✅ All embeddings from '{filename}' removed."


def rename_speaker_ui(old_name: str, new_name: str) -> str:
    if not old_name.strip() or not new_name.strip():
        return "❌ Both names required."
    from speaker_setup import cmd_rename
    class _Args:
        old = old_name.strip()
        new = new_name.strip()
    try:
        cmd_rename(_Args())
        return f"✅ Renamed '{old_name}' → '{new_name}'."
    except Exception as ex:
        return f"❌ Error: {ex}"


# ---------------------------------------------------------------------------
# Gradio UI layout
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Whisper Suite", theme=gr.themes.Soft()) as demo:

        gr.Markdown("# 🎙️ Whisper Suite")
        gr.Markdown("Dual-model transcription with speaker diarization.")

        # ── Tab 1: Transcribe ──────────────────────────────────────────────
        with gr.Tab("Transcribe"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.File(
                        label="Audio file",
                        file_types=["audio"],
                        file_count="single",
                    )
                    mode_radio = gr.Radio(
                        choices=["dual", "whisper", "whisperx"],
                        value="dual",
                        label="Mode",
                        info="dual = Whisper + WhisperX + reconcile",
                    )
                    speaker_filter_box = gr.Textbox(
                        label="Speaker filter (optional)",
                        placeholder="Alice, Bob   — blank = use all",
                        info="Comma-separated names. Restricts matching to these speakers only.",
                    )
                    cut_silence_chk = gr.Checkbox(
                        label="Cut silence (remap existing transcription)",
                        value=False,
                        info="Removes long silences, remaps timestamps. Does NOT re-transcribe.",
                    )
                    run_btn = gr.Button("▶  Start transcription", variant="primary")

                with gr.Column(scale=2):
                    log_box = gr.Textbox(
                        label="Progress log",
                        lines=12,
                        max_lines=30,
                        interactive=False,
                    )
                    status_label = gr.Label(label="Status", value="idle")

            gr.Markdown("### Segments — edit speaker names, then save")
            gr.Markdown(
                "Click any cell in the **Speaker** column to edit. "
                "Known speaker names: see dropdown. "
                "Type a new name for an unknown speaker."
            )

            known_speakers = _get_known_speakers()
            seg_table = gr.Dataframe(
                headers=["#", "Start", "End", "Speaker", "Text", "Source"],
                datatype=["number", "str", "str", "str", "str", "str"],
                col_count=(6, "fixed"),
                interactive=True,
                wrap=True,
                label="Transcript segments",
            )

            with gr.Row():
                confident_chk = gr.Checkbox(
                    label="High-confidence correction",
                    value=False,
                    info="Check if recording quality is clean — embeddings get higher weight.",
                )
                save_btn = gr.Button("💾  Save & update voiceprints", variant="secondary")

            save_status = gr.Textbox(label="Save status", interactive=False)

        # ── Tab 2: Speaker Setup ───────────────────────────────────────────
        with gr.Tab("Speaker Setup"):
            gr.Markdown("### Apply cold-start mapping")
            gr.Markdown(
                "After the first run on a new audio file, a `pending_<stem>.json` "
                "is created in `DATA_ROOT/pending_mappings/`. "
                "Open it, fill in `real_name` for each label, then apply below."
            )
            with gr.Row():
                mapping_file_input = gr.File(
                    label="pending_<stem>.json file",
                    file_types=[".json"],
                )
                audio_for_mapping  = gr.File(
                    label="Original audio (for embedding extraction)",
                    file_types=["audio"],
                )
            as_anchor_chk = gr.Checkbox(
                label="Store as gold-standard anchors (permanent)",
                value=False,
            )
            apply_mapping_btn = gr.Button("Apply mapping", variant="primary")
            mapping_status    = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("### Database summary")
            refresh_db_btn = gr.Button("Refresh")
            db_summary_box = gr.Textbox(
                label="Speakers", interactive=False, lines=8,
                value=db_summary(),
            )

            gr.Markdown("### File weight")
            with gr.Row():
                weight_file_box = gr.Textbox(label="Source filename (e.g. A.mp3)")
                weight_slider   = gr.Slider(0.0, 1.0, value=0.8, step=0.05,
                                            label="Weight")
            weight_btn    = gr.Button("Set weight")
            weight_status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("### Drop file embeddings")
            with gr.Row():
                drop_file_box = gr.Textbox(label="Source filename to drop")
            drop_btn    = gr.Button("Drop", variant="stop")
            drop_status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("### Rename speaker")
            with gr.Row():
                rename_old = gr.Textbox(label="Current name")
                rename_new = gr.Textbox(label="New name")
            rename_btn    = gr.Button("Rename")
            rename_status = gr.Textbox(label="Status", interactive=False)

        # ── Event wiring ───────────────────────────────────────────────────

        run_btn.click(
            fn=run_transcription,
            inputs=[audio_input, mode_radio, cut_silence_chk, speaker_filter_box],
            outputs=[log_box, seg_table, status_label],
        )

        save_btn.click(
            fn=save_and_update,
            inputs=[seg_table, confident_chk],
            outputs=[save_status],
        )

        apply_mapping_btn.click(
            fn=apply_pending_mapping,
            inputs=[mapping_file_input, audio_for_mapping, as_anchor_chk],
            outputs=[mapping_status],
        )

        refresh_db_btn.click(
            fn=db_summary,
            outputs=[db_summary_box],
        )

        weight_btn.click(
            fn=set_file_weight_ui,
            inputs=[weight_file_box, weight_slider],
            outputs=[weight_status],
        )

        drop_btn.click(
            fn=drop_file_ui,
            inputs=[drop_file_box],
            outputs=[drop_status],
        )

        rename_btn.click(
            fn=rename_speaker_ui,
            inputs=[rename_old, rename_new],
            outputs=[rename_status],
        )

    return demo


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

def launch(share: bool = False, port: int = 7860) -> None:
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=share,
        inbrowser=True,      # auto-open browser tab
        quiet=True,          # suppress Gradio's own startup noise
    )


if __name__ == "__main__":
    import patches
    patches.apply_all()
    launch()
