"""
ui/tab_review.py
================
Layout and event wiring for the Review tab.
Combines: audio player (top) + conflict resolver (bottom, side-by-side diff).
"""

from __future__ import annotations
import gradio as gr

from handlers import review_handlers as H


def build(demo: gr.Blocks) -> None:

    with gr.Tab("Review & Player"):

        # ── Audio player (pinned at top) ──────────────────────────────────
        gr.Markdown("## 🎵 Audio Player")
        gr.Markdown(
            "Click any row in the table below → player seeks to that segment. "
            "Keep this tab open while reviewing conflicts."
        )

        with gr.Row():
            player_audio  = gr.Audio(label="Audio", interactive=False)
            player_status = gr.Textbox(label="", interactive=False,
                                       max_lines=1, scale=1)

        refresh_player_btn = gr.Button("Refresh segment list")
        player_table = gr.Dataframe(
            headers=["Start", "End", "Speaker", "Text"],
            datatype=["str", "str", "str", "str"],
            column_count=(4, "fixed"),
            interactive=False, wrap=True, label="Click a row to seek",
        )

        gr.Markdown("---")

        # ── Conflict resolver ──────────────────────────────────────────────
        gr.Markdown("## ⚔️  Conflict Resolver")
        gr.Markdown(
            "Segments where Whisper and WhisperX disagreed. "
            "Click a row to see the side-by-side diff below."
        )

        load_conflicts_btn = gr.Button("Load conflicts from current transcript",
                                       variant="primary")
        conflict_status = gr.Textbox(label="Status", interactive=False)

        conflict_table = gr.Dataframe(
            headers=["Seg#", "Start", "End", "Speaker",
                     "[WHISPER]", "[WHISPERX]"],
            datatype=["number", "str", "str", "str", "str", "str"],
            column_count=(6, "fixed"),
            interactive=False, wrap=True, label="Conflicts",
        )

        # ── Side-by-side diff ────────────────────────────────────────────
        gr.Markdown("### Diff view  *(select a row above)*")
        diff_html = gr.HTML(
            value="<p style='color:gray'>Select a conflict row to see diff.</p>",
            label="",
        )

        # ── Resolution controls ──────────────────────────────────────────
        with gr.Row():
            seg_idx_box = gr.Textbox(
                label="Seg # to resolve", placeholder="42",
                info="Copy the Seg# from the conflict table",
            )
            choice_radio = gr.Radio(
                ["whisper", "whisperx", "keep both"],
                value="whisperx", label="Keep which version?",
            )
            speaker_box = gr.Textbox(
                label="Speaker (optional override)",
                placeholder="Alice  — blank = inherit from WhisperX",
                info="Useful when choosing Whisper version (which has no speaker info)",
            )

        resolve_btn    = gr.Button("✅  Resolve selected", variant="primary")
        resolve_status = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("---")
        gr.Markdown("### Resolve all conflicts at once")
        with gr.Row():
            resolve_all_choice = gr.Radio(
                ["whisper", "whisperx"], value="whisperx",
                label="Keep which version for all?",
            )
            resolve_all_btn = gr.Button("⚡  Resolve all", variant="stop")
        resolve_all_status = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("---")
        gr.Markdown("### ✂️  Split a merged segment (manual repair)")
        gr.Markdown(
            "If a segment contains speech from two speakers merged together, "
            "split it at a word boundary. Requires word-level alignment data."
        )
        with gr.Row():
            split_seg_box  = gr.Textbox(label="Seg # to split", placeholder="42")
            split_word_box = gr.Textbox(
                label="Split after word #",
                placeholder="5",
                info="Words are 0-indexed. Split after word N → first part gets words 0..N",
            )
            split_speaker  = gr.Textbox(
                label="Speaker for second part",
                placeholder="Bob",
            )
        split_btn    = gr.Button("✂️  Split segment")
        split_status = gr.Textbox(label="Status", interactive=False)

        # ── Event wiring ──────────────────────────────────────────────────

        # Player
        refresh_player_btn.click(fn=H.get_player_rows, outputs=[player_table])
        player_table.select(
            fn=lambda evt, audio: H.seek_to_segment(evt, audio),
            inputs=[player_audio],
            outputs=[player_audio, player_status],
        )

        # Conflict loading
        load_conflicts_btn.click(
            fn=H.load_conflicts,
            outputs=[conflict_status, conflict_table],
        )

        # Click row → update diff HTML + pre-fill seg_idx_box
        def _on_row_select(evt: gr.SelectData):
            seg_idx = str(evt.value) if evt.value is not None else ""
            # evt.index is (row, col) — we want the Seg# from col 0
            try:
                row_idx = evt.index[0]
                segs_all = __import__('handlers.state', fromlist=['get_segs']).get_segs()
                # conflict_rows returns [idx, start, end, speaker, w, wx]
                # We need to find the conflict row's seg index
                from handlers.segment_helpers import conflict_rows
                rows = conflict_rows(segs_all)
                if row_idx < len(rows):
                    seg_num = str(rows[row_idx][0])
                    html    = H.make_diff_html(seg_num)
                    return seg_num, html
            except Exception:
                pass
            return "", H.make_diff_html("")

        conflict_table.select(
            fn=_on_row_select,
            outputs=[seg_idx_box, diff_html],
        )

        resolve_btn.click(
            fn=H.resolve_one,
            inputs=[seg_idx_box, choice_radio, speaker_box],
            outputs=[resolve_status, conflict_table, diff_html],
        )
        resolve_all_btn.click(
            fn=H.resolve_all,
            inputs=[resolve_all_choice],
            outputs=[resolve_all_status],
        )
        split_btn.click(
            fn=H.split_segment,
            inputs=[split_seg_box, split_word_box, split_speaker],
            outputs=[split_status, conflict_table],
        )
