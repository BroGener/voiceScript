"""
config.py — Unified configuration.

Priority order for secrets (HF_TOKEN, DATA_ROOT):
  1. .env file in project root  (recommended — excluded from git)
  2. System environment variables
  3. Hardcoded fallbacks below   (only used if neither above is set)

To set up: copy .env.example → .env, fill in your values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Load .env (if python-dotenv is installed; graceful fallback if not)
# ---------------------------------------------------------------------------

def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=False)
    except ImportError:
        # python-dotenv not installed — read .env manually (basic parser)
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val

_load_env()


# ---------------------------------------------------------------------------
# Resolved values (env → fallback)
# ---------------------------------------------------------------------------

HF_TOKEN:   str = os.environ.get("HF_TOKEN",   "hf_YOUR_TOKEN_HERE")
AUDIO_PATH: str = os.environ.get("AUDIO_PATH", r"D:\path\to\your\audio.mp3")
DATA_ROOT:  str = os.environ.get("DATA_ROOT",  r"D:\whisper_data")


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DeviceConfig:
    device: str = "cuda"
    compute_type: str = "float16"   # cuda: "float16"; cpu: "int8" or "float32"


@dataclass
class WhisperConfig:
    model_size: str = "large-v2"
    language: str = "en"
    task: str = "transcribe"        # "transcribe" or "translate"
    verbose: bool = False           # suppressed by default; web UI shows logs instead
    fp16: bool = True


@dataclass
class WhisperXConfig:
    model_size: str = "large-v2"
    language: str = "en"
    batch_size: int = 16
    beam_size: int = 10
    compute_type: str = "float16"
    return_char_alignments: bool = False
    vad_onset: float = 0.3
    vad_offset: float = 0.3
    chunk_size_s: float = 30.0
    min_speakers: int = 1
    max_speakers: int = 5
    num_speakers: int | None = None


@dataclass
class ReconcilerConfig:
    time_tolerance_s: float = 1.5
    # "keep_both" | "prefer_whisper" | "prefer_whisperx"
    conflict_strategy: str = "keep_both"
    hallucination_repeat_threshold: int = 3


@dataclass
class SpeakerConfig:
    similarity_threshold: float = 0.80
    max_embeddings_per_speaker: int = 50
    embedding_strategy: str = "rolling"     # "rolling" | "keep_first"
    min_segment_duration_s: float = 0.8
    anchor_blend: float = 0.6
    correction_distance_threshold: float = 0.25
    remap_min_confidence: float = 0.70


@dataclass
class DiarizationPostProcessConfig:
    enabled: bool = True
    # "default" | "dialogue" | "lecture"
    scene_preset: str = "default"
    merge_gap_s: float = 0.3
    min_segment_s: float = 0.3
    overlap_strategy: str = "longer"        # "longer" | "earlier" | "keep"


@dataclass
class AudioProcessorConfig:
    silence_thresh_db: float = -35.0
    min_silence_duration_s: float = 0.8
    padding_s: float = 0.15
    output_format: str = "mp3"
    # Default False: cut + remap existing transcription (no re-transcription)
    re_transcribe_after_cut: bool = False


@dataclass
class PathConfig:
    _root: Path = field(default_factory=lambda: Path(DATA_ROOT))

    def _ensure(self, sub: str) -> Path:
        p = self._root / sub
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data_root(self) -> Path:
        return self._root

    @property
    def speakers_dir(self) -> Path:
        return self._ensure("speakers")

    @property
    def transcripts_dir(self) -> Path:
        return self._ensure("transcripts")

    @property
    def processed_audio_dir(self) -> Path:
        return self._ensure("processed_audio")

    @property
    def version_file(self) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        return self._root / "env_versions.json"

    @property
    def pending_mapping_dir(self) -> Path:
        """Directory for pending_mapping_<stem>.json files (one per audio file)."""
        return self._ensure("pending_mappings")


class AppConfig:
    """
    Global config singleton. Import everywhere via: from config import cfg
    Runtime override example: cfg.diarization_post.scene_preset = "dialogue"
    """
    def __init__(self) -> None:
        self.hf_token: str = HF_TOKEN
        self.audio_path: Path = Path(AUDIO_PATH)
        self.device: DeviceConfig = DeviceConfig()
        self.whisper: WhisperConfig = WhisperConfig()
        self.whisperx: WhisperXConfig = WhisperXConfig()
        self.reconciler: ReconcilerConfig = ReconcilerConfig()
        self.speaker: SpeakerConfig = SpeakerConfig()
        self.diarization_post: DiarizationPostProcessConfig = DiarizationPostProcessConfig()
        self.audio_processor: AudioProcessorConfig = AudioProcessorConfig()
        self.paths: PathConfig = PathConfig()


cfg = AppConfig()
