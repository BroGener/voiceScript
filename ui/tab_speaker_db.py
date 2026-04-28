"""
ui/tab_speaker_db.py
====================
Layout and event wiring for the Speaker DB tab.
"""

from __future__ import annotations
import gradio as gr

from handlers import speaker_handlers as H


def build(demo: gr.Blocks) -> None:

    with gr.Tab("Speaker DB"):

        gr.Markdown("## Speaker database management")

        with gr.Row():
            refresh_btn = gr.Button("Refresh")
        db_text = gr.Textbox(
            label="Database summary", interactive=False,
            lines=8, value=H.db_summary(),
        )

        gr.Markdown("---")
        gr.Markdown("### Apply pending speaker mapping")
        gr.Markdown(
            "After cold-start, `pending_<stem>.json` files appear in "
            "`DATA_ROOT/pending_mappings/`. Open the file, fill in `real_name` "
            "for each SPEAKER_XX, then apply below."
        )
        with gr.Row():
            pending_list        = gr.Textbox(label="Pending files",
                                             value=H.list_pending(),
                                             interactive=False)
            refresh_pending_btn = gr.Button("Refresh")

        with gr.Row():
            map_file    = gr.File(label="pending_*.json", file_types=[".json"])
            audio_4_map = gr.File(label="Original audio (for DB seeding)",
                                   file_types=["audio"])
        as_anchor_chk = gr.Checkbox(label="Store as gold-standard anchors",
                                     value=False)
        apply_btn  = gr.Button("Apply mapping", variant="primary")
        map_status = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("---")
        gr.Markdown("### File weight  *(0.0 = exclude from centroid)*")
        with gr.Row():
            w_file   = gr.Textbox(label="Source filename (e.g. A.mp3)")
            w_slider = gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="Weight")
        w_btn    = gr.Button("Set weight")
        w_status = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("### Drop all embeddings from a file")
        with gr.Row():
            d_file = gr.Textbox(label="Source filename")
        d_btn    = gr.Button("Drop", variant="stop")
        d_status = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("### Rename speaker")
        with gr.Row():
            r_old = gr.Textbox(label="Current name")
            r_new = gr.Textbox(label="New name")
        r_btn    = gr.Button("Rename")
        r_status = gr.Textbox(label="Status", interactive=False)

        # ── Event wiring ────────────────────────────────────────────────
        refresh_btn.click(fn=H.db_summary, outputs=[db_text])
        refresh_pending_btn.click(fn=H.list_pending, outputs=[pending_list])
        apply_btn.click(
            fn=H.apply_mapping,
            inputs=[map_file, audio_4_map, as_anchor_chk],
            outputs=[map_status],
        )
        w_btn.click(fn=H.set_weight, inputs=[w_file, w_slider], outputs=[w_status])
        d_btn.click(fn=H.drop_file,  inputs=[d_file],           outputs=[d_status])
        r_btn.click(fn=H.rename_speaker, inputs=[r_old, r_new], outputs=[r_status])
