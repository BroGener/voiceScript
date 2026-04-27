"""
transcriber_whisper.py
======================
原版 openai-whisper 转写器封装。
职责：加载模型、转写、将结果转换为统一 Segment 格式。
模型懒加载，首次 transcribe() 调用时才加载，避免浪费显存。
"""

from __future__ import annotations
import gc
from pathlib import Path

from config import cfg
from utils import Segment


class WhisperTranscriber:

    def __init__(self) -> None:
        self._model = None
        self._cfg = cfg.whisper
        self._dev = cfg.device

    def transcribe(self, audio_path: Path | str) -> list[Segment]:
        """
        转写音频，返回 Segment 列表。
        时间轴来自 Whisper 原始输出（比 WhisperX 更贴近真实时间）。
        """
        audio_path = Path(audio_path)
        print(f"\n[Whisper] 开始转写: {audio_path.name}")
        model = self._load_model()

        raw = model.transcribe(
            str(audio_path),
            language=self._cfg.language,
            task=self._cfg.task,
            verbose=self._cfg.verbose,
            fp16=self._cfg.fp16 and (self._dev.device == "cuda"),
        )

        segments = self._to_segments(raw["segments"])
        print(f"[Whisper] 完成，共 {len(segments)} 个片段。")
        return segments

    def unload(self) -> None:
        """主动释放显存。"""
        if self._model is not None:
            import torch
            del self._model
            self._model = None
            gc.collect()
            if self._dev.device == "cuda":
                torch.cuda.empty_cache()
            print("[Whisper] 模型已卸载。")

    def _load_model(self):
        if self._model is None:
            import whisper
            print(f"[Whisper] 加载模型 {self._cfg.model_size} ...")
            self._model = whisper.load_model(self._cfg.model_size, device=self._dev.device)
            print("[Whisper] 模型加载完毕。")
        return self._model

    @staticmethod
    def _to_segments(raw_segments: list[dict]) -> list[Segment]:
        return [
            Segment(
                start=float(s["start"]),
                end=float(s["end"]),
                text=s["text"].strip(),
                speaker="SPEAKER_UNKNOWN",
                source="whisper",
                words=s.get("words", []),
                extra={
                    "avg_logprob":        s.get("avg_logprob"),
                    "no_speech_prob":     s.get("no_speech_prob"),
                    "compression_ratio":  s.get("compression_ratio"),
                },
            )
            for s in raw_segments
        ]
