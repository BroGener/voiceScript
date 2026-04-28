"""
ui/tab_transcribe.py
====================
Layout and event wiring for the Transcribe & Edit tab.
"""

from __future__ import annotations
import gradio as gr

from handlers.segment_helpers import TABLE_HEADERS
from handlers import transcribe_handlers as H


def build(demo: gr.Blocks) -> None:
    """Adds the Transcribe & Edit tab to the Blocks context."""

    with gr.Tab("Transcribe & Edit"):

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.File(label="Audio file", file_types=["audio"])
                mode_radio  = gr.Radio(
                    ["dual", "whisper", "whisperx"], value="dual", label="Mode",
                    info="dual = Whisper + WhisperX + reconcile",
                )
                speaker_filter = gr.Textbox(
                    label="Speaker filter (optional)",
                    placeholder="Alice, Bob  — blank = all",
                )
                offset_box  = gr.Number(label="Time offset (s)", value=0.0,
                                        info="Shift all timestamps (negative = earlier)")
                cut_chk     = gr.Checkbox(label="Cut silence (remap mode)", value=False)
                run_btn     = gr.Button("▶  Transcribe", variant="primary")

            with gr.Column(scale=2):
                log_box    = gr.Textbox(label="Progress", lines=10,
                                        max_lines=20, interactive=False)
                status_lbl = gr.Label(label="Status", value="idle")

        gr.Markdown("---")
        gr.Markdown("### Load existing transcript")
        with gr.Row():
            load_file = gr.File(label="Load JSON transcript", file_types=[".json"])
            load_btn  = gr.Button("Load")
        load_status = gr.Textbox(label="Load status", interactive=False)

        gr.Markdown("---")
        gr.Markdown("### Transcript segments  *(edit Speaker or Text, then Save)*")
        seg_table = gr.Dataframe(
            headers=TABLE_HEADERS,
            datatype=["number", "str", "str", "str", "str", "str", "str"],
            column_count=(7, "fixed"),
            interactive=True, wrap=True, label="Segments",
        )
        with gr.Row():
            confident_chk = gr.Checkbox(
                label="High-confidence correction", value=False,
                info="Checked = clean recording. Higher embedding weight.",
            )
            save_btn = gr.Button("💾  Save & update voiceprints", variant="secondary")
        save_status = gr.Textbox(label="Save status", interactive=False)

        gr.Markdown("---")
        gr.Markdown("### Actions on selected rows")
        gr.Markdown("Enter row `#` numbers separated by commas, e.g. `0, 3, 7`")
        with gr.Row():
            sel_box    = gr.Textbox(label="Row indices", placeholder="0, 3, 7")
            vp_btn     = gr.Button("🧬  Extract voiceprints from selection")
            export_btn = gr.Button("📤  Export selection as SRT")
        action_status = gr.Textbox(label="Action status", interactive=False)
        export_path   = gr.Textbox(label="Exported file",  interactive=False)

        gr.Markdown("---")
        gr.Markdown("### Time offset (bulk shift)")
        with gr.Row():
            post_offset = gr.Number(label="Offset (s)", value=0.0)
            offset_btn  = gr.Button("Apply offset")
        offset_status = gr.Textbox(label="Offset status", interactive=False)

        gr.Markdown("---")
        gr.Markdown("### Silence cutting (segment-gap based)")
        gr.Markdown(
            "Cuts gaps between segments larger than the threshold. "
            "Original timestamps preserved as `[orig: ...]` annotations."
        )
        with gr.Row():
            min_gap_box = gr.Number(label="Min gap (s)", value=30.0)
            cut_pad_box = gr.Number(label="Padding (s)", value=0.15)
            cut_btn     = gr.Button("✂️  Cut silence from gaps")
        cut_status = gr.Textbox(label="Cut status", interactive=False)

        # ── Event wiring ────────────────────────────────────────────────
        run_btn.click(
            fn=H.run_transcription,
            inputs=[audio_input, mode_radio, cut_chk, speaker_filter, offset_box],
            outputs=[log_box, seg_table, status_lbl],
        )
        load_btn.click(
            fn=H.load_existing,
            inputs=[load_file],
            outputs=[load_status, seg_table],
        )
        save_btn.click(
            fn=H.save_segments,
            inputs=[seg_table, confident_chk],
            outputs=[save_status],
        )
        vp_btn.click(
            fn=H.extract_voiceprints,
            inputs=[sel_box, confident_chk],
            outputs=[action_status],
        )
        export_btn.click(
            fn=H.export_selected,
            inputs=[sel_box],
            outputs=[action_status, export_path],
        )
        offset_btn.click(
            fn=H.apply_time_offset,
            inputs=[seg_table, post_offset],
            outputs=[seg_table, offset_status],
        )
        cut_btn.click(
            fn=H.cut_silence_from_gaps,
            inputs=[min_gap_box, cut_pad_box],
            outputs=[cut_status, seg_table],
        )
