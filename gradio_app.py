"""
gradio_app.py
=============
Entry point for the Gradio web interface.
Assembles tabs from ui/ modules; all business logic lives in handlers/.

Launch:
  python main.py --web
  python gradio_app.py
"""

from __future__ import annotations
import gradio as gr

import patches
patches.apply_all()

from ui import tab_transcribe, tab_review, tab_speaker_db


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Whisper Suite") as demo:
        gr.Markdown("# 🎙️ Whisper Suite")
        tab_transcribe.build(demo)
        tab_review.build(demo)
        tab_speaker_db.build(demo)
    return demo


def launch(share: bool = False, port: int = 7860) -> None:
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=share,
        inbrowser=True,
    )


if __name__ == "__main__":
    launch()
