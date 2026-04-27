"""
transcriber_whisperx.py
=======================
WhisperX 转写器封装（ASR → 时间对齐 → 说话人分割 → 后处理）。
"""

from __future__ import annotations
import gc
from pathlib import Path

from config import cfg
from diarization_postprocessor import DiarizationPostProcessor
from utils import Segment


class WhisperXTranscriber:

    def __init__(self) -> None:
        self._asr_model    = None
        self._align_model  = None
        self._align_meta   = None
        self._diarize_model = None
        self._cfg = cfg.whisperx
        self._dev = cfg.device

    # ── 公开接口 ─────────────────────────────────────────────

    def transcribe(
        self,
        audio_path: Path | str,
        known_speakers: dict | None = None,
    ) -> list[Segment]:
        """
        完整 pipeline：ASR → 对齐 → 说话人分割 → 后处理 → 合并。

        Parameters
        ----------
        audio_path      : 音频文件路径
        known_speakers  : {speaker_name: embedding_array, ...}（预留，由 SpeakerManager 提供）
        """
        audio_path = Path(audio_path)
        print(f"\n[WhisperX] 开始转写: {audio_path.name}")

        raw      = self._run_asr(audio_path)
        aligned  = self._run_align(raw["segments"], audio_path)
        diarized = self._run_diarize(audio_path, known_speakers)

        import whisperx
        result   = whisperx.assign_word_speakers(diarized, aligned)

        segments = self._to_segments(result["segments"])
        print(f"[WhisperX] 完成，共 {len(segments)} 个片段。")
        return segments

    def get_diarize_records(self, audio_path: Path | str) -> list[dict]:
        """
        仅运行说话人分割，返回 [{speaker, start, end}, ...]。
        供 SpeakerManager 提取声纹用。
        """
        audio_path = Path(audio_path)
        annotation = self._run_diarize(audio_path)
        records = []
        try:
            for turn, _, speaker in annotation.itertracks(yield_label=True):
                records.append({"speaker": speaker, "start": turn.start, "end": turn.end})
        except AttributeError:
            pass
        return records

    def unload(self) -> None:
        for attr in ("_asr_model", "_align_model", "_diarize_model"):
            obj = getattr(self, attr, None)
            if obj is not None:
                del obj
                setattr(self, attr, None)
        self._align_meta = None
        gc.collect()
        try:
            import torch
            if self._dev.device == "cuda":
                torch.cuda.empty_cache()
        except ImportError:
            pass
        print("[WhisperX] 模型已卸载。")

    # ── 私有：各 pipeline 阶段 ────────────────────────────────

    def _load_asr(self):
        if self._asr_model is None:
            import whisperx

            # 场景预设可能覆盖 VAD 参数（dialogue / lecture / default）
            vad_overrides = DiarizationPostProcessor().get_vad_overrides() or {}
            vad_opts = {
                "chunk_size_s": self._cfg.chunk_size_s,
                "vad_onset":    self._cfg.vad_onset,
                "vad_offset":   self._cfg.vad_offset,
                **vad_overrides,
            }
            if vad_overrides:
                print(f"[WhisperX] 场景预设 '{cfg.diarization_post.scene_preset}' "
                      f"覆盖 VAD: {vad_overrides}")

            print(f"[WhisperX] 加载 ASR 模型 {self._cfg.model_size} ...")
            self._asr_model = whisperx.load_model(
                self._cfg.model_size,
                self._dev.device,
                compute_type=self._cfg.compute_type,
                asr_options={"beam_size": self._cfg.beam_size},
                vad_options=vad_opts,
            )
            print("[WhisperX] ASR 模型加载完毕。")
        return self._asr_model

    def _load_align(self):
        if self._align_model is None:
            import whisperx
            print("[WhisperX] 加载对齐模型...")
            self._align_model, self._align_meta = whisperx.load_align_model(
                language_code=self._cfg.language,
                device=self._dev.device,
            )
            print("[WhisperX] 对齐模型加载完毕。")
        return self._align_model, self._align_meta

    def _load_diarize(self):
        if self._diarize_model is None:
            import whisperx
            print("[WhisperX] 加载说话人分割模型...")
            self._diarize_model = whisperx.diarize.DiarizationPipeline(
                use_auth_token=cfg.hf_token,
                device=self._dev.device,
            )
            print("[WhisperX] 说话人分割模型加载完毕。")
        return self._diarize_model

    def _run_asr(self, audio_path: Path) -> dict:
        model = self._load_asr()
        print("[WhisperX] 语音转录中...")
        return model.transcribe(
            str(audio_path),
            batch_size=self._cfg.batch_size,
            language=self._cfg.language,
        )

    def _run_align(self, segments: list, audio_path: Path) -> dict:
        align_model, metadata = self._load_align()
        print("[WhisperX] 时间轴对齐中...")
        import whisperx
        return whisperx.align(
            segments, align_model, metadata,
            str(audio_path), self._dev.device,
            return_char_alignments=self._cfg.return_char_alignments,
        )

    def _run_diarize(self, audio_path: Path, known_speakers: dict | None = None):
        """说话人分割 + 后处理（合并碎片、解决重叠）。"""
        diarize_model = self._load_diarize()
        print("[WhisperX] 说话人识别中...")

        kwargs: dict = {}
        if self._cfg.num_speakers is not None:
            kwargs["num_speakers"] = self._cfg.num_speakers
        else:
            kwargs["min_speakers"] = self._cfg.min_speakers
            kwargs["max_speakers"] = self._cfg.max_speakers

        raw_annotation = diarize_model(str(audio_path), **kwargs)

        post = DiarizationPostProcessor()
        return post.process(raw_annotation)

    @staticmethod
    def _to_segments(raw_segments: list[dict]) -> list[Segment]:
        return [
            Segment(
                start=float(s["start"]),
                end=float(s["end"]),
                text=s.get("text", "").strip(),
                speaker=s.get("speaker", "SPEAKER_UNKNOWN"),
                source="whisperx",
                words=s.get("words", []),
            )
            for s in raw_segments
        ]
